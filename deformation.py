from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.structures import Meshes

from src.fauna.utils.skinning_v4 import estimate_bones, skinning
from src.geoaware_api import get_correspondence_features, init_geoaware_models
from src.pytorch3d_api import (
    get_mesh_with_texture_atlas,
    get_textured_mesh,
    put_obj_into_scene,
)
from src.render import get_renderer
from src.utils import load_view_points


class DeformModel(nn.Module):
    def __init__(
        self,
        scene,
        prior_shape,
        camera_view,
        init_translation,
        init_rotation,
        device="cuda",
    ):
        super().__init__()
        self.device = device
        self.scene = scene
        self.prior_shape = prior_shape

        self.bones, self.kinematic_tree, _ = self.get_prior_bones()

        self.renderer = get_renderer(*camera_view, device=self.device)
        self.num_cameras = len(camera_view[0])

        self.translation_params = nn.Parameter(torch.tensor(init_translation))
        self.rotation_params = nn.Parameter(torch.tensor(init_rotation))
        self.scale_params = nn.Parameter(torch.tensor([0.3] * 3))
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
        return put_obj_into_scene(
            deformed_mesh,
            self.scene,
            R=self.get_rotation_params(),
            T=self.get_translation_params(),
            S=self.get_scale_params(),
        )

    def forward(self):
        verts_articulated, aux = skinning(
            self.prior_shape.verts_packed()[None],
            self.bones,
            self.kinematic_tree,
            self.arti_params[None, None],
            output_posed_bones=True,
            temperature=0.05,
        )

        verts_articulated = verts_articulated.squeeze(0, 1)
        deformed_mesh = Meshes(
            [verts_articulated],
            [self.prior_shape.faces_packed()],
            self.prior_shape.textures,
        )

        concat_mesh = self.forward_affordance(deformed_mesh)

        clone_meshes = concat_mesh.extend(self.num_cameras)

        images = self.renderer(clone_meshes)

        return images


def train(model, reference_image, cfg):

    geoaware_models = init_geoaware_models(
        aggre_net_path=cfg.geoaware_cfg.aggre_net_path
    )

    for i in range(1):
        images = model()

        images = images[..., :3].squeeze(0).permute(2, 0, 1)

        model.cpu()
        torch.cuda.empty_cache()

        get_correspondence_features(images, *geoaware_models)


def main(cfg, scene_cfg):
    device = torch.device("cuda:0")

    scene = get_textured_mesh(scene_cfg.scene.mesh_path, device=device)

    scene_name = cfg.data.scenes[cfg.scene_idx]
    prompt = scene_cfg.prompt_ids[cfg.prompt_idx]

    image_dir = Path(cfg.log_dir) / scene_name / prompt
    image_dir.mkdir(parents=True, exist_ok=True)

    view_points, look_at = load_view_points(scene_name, prompt, cfg.view_points_path)

    R, T = look_at_view_transform(
        eye=torch.tensor(view_points), at=[look_at], up=((0, 0, 1),)
    )
    mesh = get_mesh_with_texture_atlas(cfg.fauna.prior_shape_path)

    deform_model = DeformModel(
        scene=scene,
        prior_shape=mesh,
        camera_view=(R[2][None], T[2][None]),
        init_rotation=[1.7453293, 0.0, -1.5707963],
        init_translation=np.array(look_at) + [-0.4, -0.9, 0.6],
        device=device,
    )
    deform_model.to(device)

    reference_image = Image.open(
        "data/log_genzi/quintyn-glenn-city-scene-kyoto/walk-bridge-1/view_0.png"
    )

    train(deform_model, reference_image, cfg)
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
