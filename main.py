from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch3d.io import save_obj
from pytorch3d.renderer import look_at_view_transform

from src.fauna_api import generate_fauna
from src.pytorch3d_api import get_textured_mesh, put_obj_into_scene
from src.render import render
from src.utils import fix_seed, load_view_points, omegaconf_to_dotdict


def main(cfg, scene_cfg):
    device = torch.device("cuda:0")

    scene_name = cfg["data.scenes"][cfg["scene_idx"]]
    prompt = scene_cfg["prompt_ids"][cfg["prompt_idx"]]

    image_dir = Path(cfg["log_dir"]) / scene_name / prompt
    image_dir.mkdir(parents=True, exist_ok=True)

    mesh = get_textured_mesh(scene_cfg["scene.mesh_path"], device=device)

    fauna = get_textured_mesh(cfg["fauna_cfg.prior_shape_path"], device=device)
    
    fauna.verts_list()[0] *= 1e-3
    # fauna = generate_fauna(cfg, device)
    # verts, faces = fauna.get_mesh_verts_faces(0)

    view_points, look_at = load_view_points(scene_name, prompt, cfg["view_points_path"])

    R, T = look_at_view_transform(
        eye=torch.tensor(view_points), at=[look_at], up=((0, 0, 1),)
    )

    concat_mesh = put_obj_into_scene(
        fauna,
        mesh,
        R=[0.0, 0.0, -90.0],
        T=np.array(look_at) + [0.4, -0.9, -0.6],
        S=[1],
        degrees=True,
    )

    render(
        concat_mesh,
        R,
        T,
        pt_light_position=(np.array(look_at) + [0, 0, 5]).tolist(),
        image_dir=image_dir,
        image_size=cfg["render.image_size"],
        device=device,
    )


if __name__ == "__main__":
    cfg_cli = OmegaConf.from_cli()
    cfg = omegaconf_to_dotdict(
        OmegaConf.merge(
            OmegaConf.load(cfg_cli.run_cfg),
            cfg_cli,
        )
    )
    scene_name = cfg["data.scenes"][cfg["scene_idx"]]
    scene_cfg_path = Path(cfg["data.root_dir"]) / (scene_name + cfg["data.cfg_suffix"])

    scene_cfg = omegaconf_to_dotdict(OmegaConf.load(scene_cfg_path.as_posix()))

    fix_seed(cfg["seed"])
    main(cfg, scene_cfg)
