from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesAtlas

from src.fauna.utils.skinning_v4 import estimate_bones, skinning
from src.geoaware_api import get_correspondence_features, init_geoaware_models, get_processed_features
from src.pytorch3d_api import (
    get_mesh_with_texture_atlas,
    get_textured_mesh,
    put_obj_into_scene,
    get_transform_function,
)
from src.render import get_renderer
from src.utils import load_view_points
from torchvision import transforms
import cv2
import numpy as np

COLOR_LIST = colors = [
    (193, 182, 255),  # Light Pink
    (235, 206, 135),  # Sky Blue
    (180, 105, 255),  # Hot Pink
    (230, 216, 173),  # Light Blue
    (213, 239, 255),  # Papaya Whip
    (140, 230, 240),  # Khaki
    (152, 251, 152),  # Pale Green
    (196, 228, 255),  # Bisque
    (122, 160, 255),  # Light Salmon
    (173, 222, 255),  # Navajo White
    (170, 232, 238),  # Pale Goldenrod
    (181, 228, 255),  # Moccasin
    (205, 250, 255),  # Lemon Chiffon
    (220, 245, 245),  # Beige
    (128, 128, 240),  # Light Coral
    (185, 218, 255),  # Peach Puff
    (203, 192, 255),  # Pink
    (221, 160, 221),  # Plum
    (47, 255, 173),   # Green Yellow
    (255, 255, 224)   # Light Cyan
]

class DeformModel(nn.Module):
    def __init__(
        self,
        scene,
        prior_shape,
        camera_view,
        init_translation,
        init_rotation,
        init_scale=1.0,
        image_size=256,
        device="cuda",
    ):
        super().__init__()
        self.device = device
        self.scene = scene
        self.prior_shape = prior_shape
        self.image_size = image_size

        self.bones, self.kinematic_tree, _ = self.get_prior_bones()
        
        # bone information
        self.num_body_bones = 8
        self.num_leg_bones = 3
        self.num_legs = 4
        self.max_arti_angle = 60
        self.reg_body_rotate_mult = 0.1

        self.renderer, self.cameras = get_renderer(*camera_view, device=self.device, image_size=self.image_size)
        self.num_cameras = len(camera_view[0])

        self.translation_params = nn.Parameter(torch.tensor(init_translation))
        self.rotation_params = nn.Parameter(torch.tensor(init_rotation))
        self.scale_params = nn.Parameter(torch.tensor([init_scale] * 3))
        self.arti_params = nn.Parameter(torch.zeros((20, 3)))

    def get_prior_bones(self):
        verts = self.prior_shape.verts_packed()

        return estimate_bones(
            verts.detach()[None, None],
            n_body_bones=8,
            n_legs=4,
            n_leg_bones=3,
            body_bones_type="z_minmax_y+",
            compute_kinematic_chain=True,
            attach_legs_to_body=True,
            bone_y_threshold=0.4,
            body_bone_idx_preset=[0, 0, 0, 0],
        )

    def get_scale_params(self):
        return self.scale_params

    def get_translation_params(self):
        return self.translation_params

    def get_rotation_params(self):
        return self.rotation_params

    def get_arti_params(self):
        return self.arti_params
    
    def forward_affordance(self, deformed_mesh):
        trans_func = get_transform_function(
            self.get_rotation_params(),
            self.get_translation_params(),
            self.get_scale_params(),
            device=self.device,
        )

        obj_v = trans_func.transform_points(deformed_mesh.verts_packed())
        obj_f = deformed_mesh.faces_packed() + self.scene.verts_packed().shape[0]

        verts = torch.cat([self.scene.verts_packed(), obj_v])
        faces = torch.cat([self.scene.faces_packed(), obj_f])
        text = TexturesAtlas(
            [torch.cat([self.scene.textures.atlas_packed(), deformed_mesh.textures.atlas_packed()])]
        )

        image_verts_2d = self.cameras.transform_points_screen(obj_v, image_size=(self.image_size, self.image_size))[:, :2]

        return Meshes(verts=[verts], faces=[faces], textures=text), image_verts_2d
    
    def get_restricted_articulation_parameters(self, articulation_angles: torch.nn.Parameter):
        articulation_angles = articulation_angles.tanh()
        # constrain_legs:
        leg_bones_posx = [
            self.num_body_bones + i
            for i in range(self.num_leg_bones * self.num_legs // 2)
        ]
        leg_bones_negx = [
            self.num_body_bones + self.num_leg_bones * self.num_legs // 2 + i
            for i in range(self.num_leg_bones * self.num_legs // 2)
        ]

        tmp_mask = torch.zeros_like(articulation_angles).to(articulation_angles.device)
        tmp_mask[:, :, leg_bones_posx + leg_bones_negx, 2] = 1
        articulation_angles = (
            tmp_mask * (articulation_angles * 0.3)
            + (1 - tmp_mask) * articulation_angles
        )  # no twist

        tmp_mask = torch.zeros_like(articulation_angles).to(articulation_angles.device)
        tmp_mask[:, :, leg_bones_posx + leg_bones_negx, 1] = 1
        articulation_angles = (
            tmp_mask * (articulation_angles * 0.3)
            + (1 - tmp_mask) * articulation_angles
        )  # (-0.4, 0.4),  limit side bending

        # new regularizations, for bottom 2 bones of each leg, they can only rotate around x-axis,
        # and for the toppest bone of legs, restrict its angles in a smaller range
        # forbid_leg_rotate
        # small_leg_angle
        # regularize the rotation angle of first leg bones
        leg_bones_top = [8, 11, 14, 17]
        # leg_bones_top = [10, 13, 16, 19]
        tmp_mask = torch.zeros_like(articulation_angles).to(articulation_angles.device)
        tmp_mask[:, :, leg_bones_top, 1] = 1
        tmp_mask[:, :, leg_bones_top, 2] = 1
        articulation_angles = (
            tmp_mask * (articulation_angles * 0.05)
            + (1 - tmp_mask) * articulation_angles
        )

        leg_bones_bottom = [9, 10, 12, 13, 15, 16, 18, 19]
        # leg_bones_bottom = [8, 9, 11, 12, 14, 15, 17, 18]
        tmp_mask = torch.ones_like(articulation_angles).to(articulation_angles.device)
        tmp_mask[:, :, leg_bones_bottom, 1] = 0
        tmp_mask[:, :, leg_bones_bottom, 2] = 0
        # tmp_mask[:, :, leg_bones_bottom, 0] = 0.3
        articulation_angles = tmp_mask * articulation_angles

        articulation_angles = articulation_angles * self.max_arti_angle / 180 * np.pi

        # check if regularize the leg-connecting body bones z-rotation first
        # then check if regularize all the body bones z-rotation
        # regularize z-rotation using 0.1 in pi-space
        body_rotate_mult = (
            self.reg_body_rotate_mult * 180 * 1.0 / (self.max_arti_angle * np.pi)
        )  # the max angle = mult*original_max_angle

        # all bones
        body_bones_mask = [0, 1, 2, 3, 4, 5, 6, 7]
        tmp_body_mask = torch.zeros_like(articulation_angles)
        tmp_body_mask[:, :, body_bones_mask, 2] = 1
        articulation_angles = (
            tmp_body_mask * (articulation_angles * body_rotate_mult)
            + (1 - tmp_body_mask) * articulation_angles
        )
        
        return articulation_angles
        
    def forward(self):
        
        articulation_params = self.get_restricted_articulation_parameters(self.arti_params[None, None])
        
        verts_articulated, aux = skinning(
            self.prior_shape.verts_packed()[None],
            self.bones,
            self.kinematic_tree,
            articulation_params,
            output_posed_bones=False,
            temperature=0.05,
        )

        verts_articulated = verts_articulated.squeeze(0, 1)

        trans_func = get_transform_function(
            self.get_rotation_params(),
            self.get_translation_params(),
            self.get_scale_params(),
            device=self.device,
        )

        verts_articulated = trans_func.transform_points(verts_articulated)


        deformed_mesh = Meshes(
            [verts_articulated],
            [self.prior_shape.faces_packed()],
            self.prior_shape.textures,
        )
        image_verts_2d = self.cameras.transform_points_screen(verts_articulated, image_size=(self.image_size, self.image_size))[:, :2]

        # concat_mesh, image_verts_2d = self.forward_affordance(deformed_mesh)

        # clone_meshes = concat_mesh.extend(self.num_cameras)
        clone_meshes = deformed_mesh.extend(self.num_cameras)

        images = self.renderer(clone_meshes)

        return images, image_verts_2d

def dot_image(image, point, color=(0, 0, 255)):
    cv2.circle(image, (int(point[0].item()), int(point[1].item())), radius=2, color=color, thickness=-1)

class MaskedRMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.eps = eps
        
    def forward(self, source, target, mask):
        loss = torch.sqrt(self.mse(source, target)[mask].sum(dim=1)).mean()
        return loss

def train(model, optimizers, reference_image, cfg, scene_device="cuda:0", geoaware_device="cuda:0", num_patches=60):

    geoaware_models = init_geoaware_models(
        aggre_net_path=cfg.geoaware_cfg.aggre_net_path,
        num_patches=num_patches,
        geoaware_device=geoaware_device,
    )
    
    rng = np.random.default_rng(seed=1102)
    criterion = MaskedRMSELoss()
    for i in range(1000):
        source_images, images_verts_2d = model()
        
        with torch.no_grad():
            source_pil_images = Image.fromarray((255 * source_images[..., :3].squeeze(0).detach().cpu().numpy()).astype("uint8"))
            source_feat = get_processed_features(source_pil_images, *geoaware_models)
            target_feat = get_processed_features(reference_image, *geoaware_models)
            
            srcfeat = nn.Upsample(size=(source_images.shape[1], source_images.shape[1]), mode="bilinear")(source_feat)
            tgtfeat = nn.Upsample(size=(reference_image.size[1], reference_image.size[1]), mode="bilinear")(target_feat)
            
            # render image
            src_output_image = cv2.cvtColor(np.array(source_pil_images), cv2.COLOR_RGB2BGR)
            tgt_output_image = cv2.cvtColor(np.array(reference_image), cv2.COLOR_RGB2BGR)
            correspondence_path = Path(f"res/corr/iteration_{i}.png")
            correspondence_path.parent.mkdir(parents=True, exist_ok=True)
            
            images_verts_2d_int = images_verts_2d.detach().cpu().numpy().astype(int)
                        
            _, unique_index = np.unique(images_verts_2d_int, return_index=True, axis=0)
            batch_size = 20
            threshold = 0.8
            
            
            
            # get <batch-size> random sample points and check if it is out of bound
            rng.shuffle(unique_index)
            src_vec, random_sample_index = [], []
   
            for index in unique_index:
                if any(images_verts_2d[index] < 0) or any(images_verts_2d[index] > source_images.shape[1]):
                    print("out of bound, select another index")
                    continue
                random_sample_index.append(index)
                x, y = images_verts_2d[index]
                src_vec.append(srcfeat[0, :, y.long().item(), x.long().item()].reshape(1, -1, 1, 1))
                if len(src_vec) == batch_size:
                    break
            
            # calculate the corresponding target point
            src_vec = torch.cat(src_vec, dim=0)
            cos_map = nn.CosineSimilarity(dim=1)(src_vec, tgtfeat).reshape(batch_size, -1)
            
            max_value, max_index = cos_map.max(dim=1)
        
            max_index_x, max_index_y = max_index // source_images.shape[1], max_index % source_images.shape[1]
        
            max_index_2d = torch.vstack([max_index_y, max_index_x]).T.float()
            
            # plot the correspondece
            src_tmp_image = src_output_image.copy()
            t2_image = tgt_output_image.copy()
            for color, (x, y), (max_x, max_y) in zip(COLOR_LIST, images_verts_2d_int[random_sample_index], max_index_2d):
                dot_image(src_tmp_image, (x, y), color=color)
                dot_image(t2_image, (max_x, max_y), color=color)
            cat_image = cv2.hconcat([src_tmp_image, t2_image])
            cv2.imwrite(correspondence_path.as_posix(), cat_image)

        # calculate loss and update model
        loss = criterion(images_verts_2d[random_sample_index], max_index_2d, mask=max_value > threshold)
        
        for optimizer in optimizers:
            optimizer.zero_grad()
        
        loss.backward()
        
        for optimizer in optimizers:
            optimizer.step()
            
        print(loss.item())
        # breakpoint()
        # for i, (x, y) in enumerate(torch.unique(images_verts_2d.int(), dim=0)):
        #     src_vec = srcfeat[0, :, y, x].reshape(1, -1, 1, 1)
        #     cos_map = nn.CosineSimilarity(dim=1)(src_vec, tgtfeat)
            
        #     max_cos = cos_map.max()

        #     res_path = Path(f"res/{i}_{max_cos:.2f}.png")
        #     res_path.parent.mkdir(parents=True, exist_ok=True)
            
        #     src_tmp_image = src_output_image.copy()
        #     t2_image = tgt_output_image.copy()
        #     dot_image(src_tmp_image, (x, y))
        #     max_yx = np.unravel_index(cos_map[0].detach().cpu().numpy().argmax(), cos_map[0].shape)
        #     dot_image(t2_image, (max_yx[1], max_yx[0]))
            
        #     cat_image = cv2.hconcat([src_tmp_image, t2_image])
        #     cv2.imwrite(res_path.as_posix(), cat_image)
            # print(f"save image {i}, confidence: {cos_map[0][max_yx]}")
        # loss = torch.nn.MSELoss()(source_feat, target_feat)
        # breakpoint()
        
        




def main(cfg, scene_cfg):
    scene_device = torch.device("cuda:0")
    geoaware_device = torch.device("cuda:0")
    # scene = get_textured_mesh(scene_cfg.scene.mesh_path, device=scene_device)

    scene_name = cfg.data.scenes[cfg.scene_idx]
    prompt = scene_cfg.prompt_ids[cfg.prompt_idx]

    image_dir = Path(cfg.log_dir) / scene_name / prompt
    image_dir.mkdir(parents=True, exist_ok=True)

    view_points, look_at = load_view_points(scene_name, prompt, cfg.view_points_path)

    # R, T = look_at_view_transform(
    #     eye=torch.tensor(view_points), at=[[look_at]], up=((0, 0, 1),)
    # )
    R, T = look_at_view_transform(eye=torch.tensor(view_points), at=[[0, 0, 0]], up=((0, 0, 1),))
    mesh = get_textured_mesh(cfg.fauna.prior_shape_path)
    mesh.verts_list()[0] *= 3e-3
    
    # deform_model = DeformModel(
    #     scene=scene,
    #     prior_shape=mesh,
    #     camera_view=(R[2][None], T[2][None]),
    #     init_rotation=[0.0, 0.0, -1.5707963],
    #     init_translation=np.array(look_at) + [0.4, -0.9, -0.6],
    #     device=scene_device,
    # )
    deform_model = DeformModel(
        scene=None,
        prior_shape=mesh,
        camera_view=(R[2][None], T[2][None]),
        init_rotation=[0, 0, -80/180*np.pi],
        init_translation=[0, 0, -2.],
        device=scene_device,
    )
    deform_model.to(scene_device)

    reference_image = Image.open(cfg.reference_image_path)
    
    optimizers = [
        torch.optim.AdamW([deform_model.get_translation_params()], lr=1e-2),
        torch.optim.AdamW([deform_model.get_rotation_params()], lr=1e-2),
        torch.optim.AdamW([deform_model.get_scale_params()], lr=1e-2),
        torch.optim.AdamW([deform_model.get_arti_params()], lr=1e-2),
    ]

    train(deform_model, optimizers, reference_image, cfg, scene_device=scene_device, geoaware_device=geoaware_device)
    # store_images(images, image_dir)


if __name__ == "__main__":
    cfg_cli = OmegaConf.from_cli()
    cfg = OmegaConf.merge(
        OmegaConf.load(cfg_cli.run_cfg),
        cfg_cli,
    )
    scene_name = cfg.data.scenes[cfg.scene_idx]
    scene_cfg_path = Path(cfg.data.root_dir) / (scene_name + cfg.data.cfg_suffix)
    scene_cfg = OmegaConf.load(scene_cfg_path.as_posix())

    main(cfg, scene_cfg)
