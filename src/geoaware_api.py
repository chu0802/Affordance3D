import torch
import torch.nn.functional as F
from odise.modeling.wrapper import OpenPanopticInference
from PIL import Image
from torchvision.transforms import Resize
import torch.distributed.pipeline.sync as sync
from .geoaware.model_utils.extractor_dino import ViTExtractor
from .geoaware.model_utils.extractor_sd import (
    build_demo_classes_and_metadata,
    load_model,
    process_features_and_mask,
)
from .geoaware.model_utils.projection_network import AggregationNetwork
from .geoaware.utils.utils_correspondence import resize
import os
# def resize(img, size=224):
#     return Resize((size, size))(img)


def get_sd_features(img, sd_model, num_patches=60):
    img_sd_input = Resize(num_patches * 16)(img)
    height, width = img_sd_input.shape[-2:]

    features_sd = sd_model.get_features(
        [{"image": img_sd_input, "height": height, "width": width}], pca=True
    )
    
    del features_sd["s2"]
    
    return features_sd

def get_dino_features(img, extractor_vit, num_patches=60):
    img_dino_input = Resize(num_patches * 14)(img)
    
    img_batch = (extractor_vit.preprocess_torch_image(img_dino_input))

    features_dino = extractor_vit.extract_descriptors(
        img_batch, layer=11, facet="token"
    )
    features_dino = features_dino.permute(0, 1, 3, 2).reshape(
        1, -1, num_patches, num_patches
    )
    
    return features_dino

def _get_correspondence_features(
    img, sd_model, aggre_net, extractor_vit, num_patches=60
): 
    features_sd = get_sd_features(img, sd_model, num_patches=num_patches)
    features_dino = get_dino_features(img, extractor_vit, num_patches)

    desc_gathered = torch.cat(
        [
            features_sd["s3"],
            F.interpolate(
                features_sd["s4"],
                size=(num_patches, num_patches),
                mode="bilinear",
                align_corners=False,
            ),
            F.interpolate(
                features_sd["s5"],
                size=(num_patches, num_patches),
                mode="bilinear",
                align_corners=False,
            ),
            features_dino,
        ],
        dim=1,
    )

    desc = aggre_net(desc_gathered)  # 1, 768, 60, 60
    
    norms_desc = torch.linalg.norm(desc, dim=1, keepdim=True)
    desc = desc / (norms_desc + 1e-8)
    return desc


def get_correspondence_features(img, sd_model, aggre_net, extractor_vit, num_patches=60):
    with torch.no_grad():
        return _get_correspondence_features(img, sd_model, aggre_net, extractor_vit, num_patches)
    

def get_processed_features(
    img, sd_model, sd_aug, aggre_net, extractor_vit, num_patches=60
):
    # extract stable diffusion features
    img_sd_input = resize(img, target_res=num_patches * 16, resize=True, to_pil=True)
    features_sd = process_features_and_mask(
        sd_model, sd_aug, img_sd_input, mask=False, raw=True
    )
    del features_sd["s2"]

    # extract dinov2 features
    img_dino_input = resize(img, target_res=num_patches * 14, resize=True, to_pil=True)
    img_batch = (extractor_vit.preprocess_pil(img_dino_input)).cuda()
    features_dino = extractor_vit.extract_descriptors(
        img_batch, layer=11, facet="token"
    )
    features_dino = features_dino.permute(0, 1, 3, 2).reshape(
        1, -1, num_patches, num_patches
    )

    # aggregate the features and apply post-processing
    desc_gathered = torch.cat(
        [
            features_sd["s3"],
            F.interpolate(
                features_sd["s4"],
                size=(num_patches, num_patches),
                mode="bilinear",
                align_corners=False,
            ),
            F.interpolate(
                features_sd["s5"],
                size=(num_patches, num_patches),
                mode="bilinear",
                align_corners=False,
            ),
            features_dino,
        ],
        dim=1,
    )
    desc = aggre_net(desc_gathered)  # 1, 768, 60, 60

    # normalize the descriptors
    norms_desc = torch.linalg.norm(desc, dim=1, keepdim=True)
    desc = desc / (norms_desc + 1e-8)
    return desc


def init_geoaware_models(aggre_net_path="results_spair/best_856.PTH", num_patches=60, geoaware_device="cuda:1"):
    sd_model, sd_aug = load_model(
        diffusion_ver="v1-5", image_size=num_patches * 16, num_timesteps=50, device=geoaware_device
    )
    extractor_vit = ViTExtractor("dinov2_vitb14", stride=14, device=geoaware_device)
    aggre_net = AggregationNetwork(
        feature_dims=[640, 1280, 1280, 768], projection_dim=768, device=geoaware_device
    )
    aggre_net.load_pretrained_weights(torch.load(aggre_net_path))

    return sd_model, sd_aug, aggre_net, extractor_vit

if __name__ == "__main__":
    num_patches = 60
    device = torch.device("cuda:0")
    aggre_net_path = "/home/chu980802/geoaware/results_spair/best_856.PTH"
    img = Image.open("/home/chu980802/Affordance3D/reference_image.png")
    
    sd_model, sd_aug = load_model(
        diffusion_ver="v1-5", image_size=num_patches * 16, num_timesteps=50, device=device
    )
    extractor_vit = ViTExtractor("dinov2_vitb14", stride=14, device=device)
    aggre_net = AggregationNetwork(
        feature_dims=[640, 1280, 1280, 768], projection_dim=768, device=device
    )
    aggre_net.load_pretrained_weights(torch.load(aggre_net_path))
    
    get_processed_features(img, sd_model, sd_aug, aggre_net, extractor_vit, num_patches)
