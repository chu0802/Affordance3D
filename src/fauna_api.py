from typing import Any, Dict, Tuple

import torch
import torchvision
from PIL import Image

from src.fauna.model_ddp import Unsup3DDDP
from src.fauna.render.mesh import Mesh
from src.fauna.utils.misc import load_yaml
from src.fauna.render.obj import write_obj


def expand2square(pil_img: Image, background_color: tuple):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def preprocess_image(input_image: Image):
    RES = 1024
    input_image.thumbnail([RES, RES], Image.Resampling.LANCZOS)
    input_image = expand2square(input_image, (0, 0, 0, 255))
    return input_image.resize((256, 256), Image.Resampling.LANCZOS)


def load_fauna_model_config(model_config_path: str):
    ## Load config
    cfgs = load_yaml(model_config_path)

    cfgs["config"] = model_config_path
    cfgs["body_bon_idx_preset"] = [3, 6, 6, 3]
    return cfgs


def load_fauna_model(
    model_path: str, model_config_path: str, device: str = "cuda"
) -> Tuple[Unsup3DDDP, torch.Tensor, torch.Tensor]:
    ## Load config
    model_cfgs = load_fauna_model_config(model_config_path)
    model = Unsup3DDDP(model_cfgs)

    model.netPrior.classes_vectors = torch.nn.Parameter(
        torch.nn.init.uniform_(torch.empty(123, 128), a=-0.05, b=0.05)
    )
    cp = torch.load(model_path, map_location=device)
    model.load_model_state(cp)
    memory_bank_keys = cp["memory_bank_keys"]
    memory_bank = cp["memory_bank"]

    model.to(device)
    memory_bank.to(device)
    memory_bank_keys.to(device)
    return (model, memory_bank, memory_bank_keys)


def get_bank_embedding(
    rgb: torch.Tensor,
    memory_bank_keys: torch.Tensor,
    memory_bank: torch.Tensor,
    model: Unsup3DDDP,
    memory_bank_topk: int = 10,
    memory_bank_dim: int = 128,
):
    images = rgb
    batch_size, num_frames, _, h0, w0 = images.shape
    images = images.reshape(batch_size * num_frames, *images.shape[2:])  # 0~1
    images_in = images * 2 - 1  # rescale to (-1, 1) for DINO

    x = images_in
    with torch.no_grad():
        b, c, h, w = x.shape
        model.netInstance.netEncoder._feats = []
        model.netInstance.netEncoder._register_hooks([11], "key")
        # self._register_hooks([11], 'token')
        x = model.netInstance.netEncoder.ViT.prepare_tokens(x)
        # x = self.ViT.prepare_tokens_with_masks(x)

        for blk in model.netInstance.netEncoder.ViT.blocks:
            x = blk(x)
        out = model.netInstance.netEncoder.ViT.norm(x)
        model.netInstance.netEncoder._unregister_hooks()

        ph, pw = (
            h // model.netInstance.netEncoder.patch_size,
            w // model.netInstance.netEncoder.patch_size,
        )
        patch_out = out[:, 1:]  # first is class token
        patch_out = patch_out.reshape(
            b, ph, pw, model.netInstance.netEncoder.vit_feat_dim
        ).permute(0, 3, 1, 2)

        patch_key = model.netInstance.netEncoder._feats[0][
            :, :, 1:
        ]  # B, num_heads, num_patches, dim
        patch_key = patch_key.permute(0, 1, 3, 2).reshape(
            b, model.netInstance.netEncoder.vit_feat_dim, ph, pw
        )

        global_feat = out[:, 0]

    batch_features = global_feat

    batch_size = batch_features.shape[0]

    query = torch.nn.functional.normalize(
        batch_features.unsqueeze(1), dim=-1
    )  # [B, 1, d_k]
    key = torch.nn.functional.normalize(memory_bank_keys, dim=-1)  # [size, d_k]
    key = (
        key.transpose(1, 0).unsqueeze(0).repeat(batch_size, 1, 1).to(query.device)
    )  # [B, d_k, size]

    cos_dist = torch.bmm(query, key).squeeze(1)  # [B, size], larger the more similar
    rank_idx = torch.sort(cos_dist, dim=-1, descending=True)[1][
        :, :memory_bank_topk
    ]  # [B, k]
    value = (
        memory_bank.unsqueeze(0).repeat(batch_size, 1, 1).to(query.device)
    )  # [B, size, d_v]

    out = torch.gather(
        value, dim=1, index=rank_idx[..., None].repeat(1, 1, memory_bank_dim)
    )  # [B, k, d_v]

    weights = torch.gather(cos_dist, dim=-1, index=rank_idx)  # [B, k]
    weights = (
        torch.nn.functional.normalize(weights, p=1.0, dim=-1)
        .unsqueeze(-1)
        .repeat(1, 1, memory_bank_dim)
    )  # [B, k, d_v] weights have been normalized

    out = weights * out
    out = torch.sum(out, dim=1)

    batch_mean_out = torch.mean(out, dim=0)

    weight_aux = {
        "weights": weights[:, :, 0],  # [B, k], weights from large to small
        "pick_idx": rank_idx,  # [B, k]
    }

    batch_embedding = batch_mean_out
    embeddings = out
    weights = weight_aux

    bank_embedding_model_input = [batch_embedding, embeddings, weights]

    return bank_embedding_model_input


def generate_fauna(cfg: Dict[str, Any], device: str = "cuda"):
    total_iter, epoch = 999999, 999

    image = Image.open(cfg["fauna_cfg.input_image"]).convert("RGB")
    input_image = preprocess_image(image)
    input_image = torch.stack(
        [torchvision.transforms.ToTensor()(input_image)], dim=0
    ).to(device)

    model, memory_bank, memory_bank_keys = load_fauna_model(
        cfg["fauna_cfg.model_path"], cfg["fauna_cfg.model_config_path"], device=device
    )

    with torch.no_grad():
        model.netPrior.eval()
        model.netInstance.eval()

        input_image = torch.nn.functional.interpolate(
            input_image, size=(256, 256), mode="bilinear", align_corners=False
        )
        input_image = input_image[:, None, :, :]  # [B=1, F=1, 3, 256, 256]

        bank_embedding = get_bank_embedding(
            input_image,
            memory_bank_keys,
            memory_bank,
            model,
            memory_bank_topk=10,
            memory_bank_dim=128,
        )

        prior_shape, dino_pred, classes_vectors = model.netPrior(
            category_name="tmp",
            perturb_sdf=False,
            total_iter=total_iter,
            is_training=False,
            class_embedding=bank_embedding,
        )

        Instance_out = model.netInstance(
            "tmp",
            input_image,
            prior_shape,
            epoch,
            dino_features=None,
            dino_clusters=None,
            total_iter=total_iter,
            is_training=False,
        )
        
        (
            shape,
            pose_raw,
            pose,
            mvp,
            w2c,
            campos,
            texture,
            im_features,
            dino_feat_im_calc,
            deformation,
            arti_params,
            light,
            forward_aux,
        ) = Instance_out
        
        class_vector = None
        if classes_vectors is not None:
            if len(classes_vectors.shape) == 1:
                class_vector = classes_vectors
        (
            image_pred,
            mask_pred,
            flow_pred,
            dino_feat_im_pred,
            albedo,
            shading,
        ) = model.render(
            shape,
            texture,
            mvp,
            w2c,
            campos,
            (256, 256),
            background=model.background_mode,
            im_features=im_features,
            light=light,
            prior_shape=prior_shape,
            render_flow = False,
            dino_pred=dino_pred,
            class_vector=class_vector[None, :].expand(1, -1),
            num_frames=1,
            spp=model.renderer_spp,
            im_features_map=None,
        )  # the real rendering process        
        # write_obj(folder=".", fname="test.obj", mesh=shape, idx=0, save_material=True, feat=None)

        # Image.fromarray((image_pred[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')).save("output_iamge.png")

        return shape.to_pytorch3d()
