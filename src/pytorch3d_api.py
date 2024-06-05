import torch
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.renderer import TexturesAtlas

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.transforms import RotateAxisAngle, Scale, Translate


def get_mesh_with_texture_atlas(filename, device="cuda"):
    meshes = load_objs_as_meshes([filename], device=device)
    verts = meshes.verts_packed()
    faces = meshes.faces_packed()
    text = TexturesAtlas([torch.ones(faces.shape[0], 1, 1, 3).to(device)])
    return Meshes(verts=[verts], faces=[faces], textures=text)


def put_obj_into_scene(obj: Meshes, scene: Meshes, R, T, S):
    obj_v = obj.verts_packed().clone()
    obj_f = obj.faces_packed().clone()

    obj_f += scene.verts_packed().shape[0]

    transforms = (
        [Scale(S, device=scene.device)]
        + [
            RotateAxisAngle(angle, axis, device=scene.device)
            for angle, axis in zip(R, ["X", "Y", "Z"])
        ]
        + [Translate(*T, device=scene.device)]
    )

    obj_v = transforms[0].compose(*transforms[1:]).transform_points(obj_v)

    verts = torch.cat([scene.verts_packed(), obj_v])
    faces = torch.cat([scene.faces_packed(), obj_f])
    text = TexturesAtlas(
        [torch.cat([scene.textures.atlas_packed(), obj.textures.atlas_packed()])]
    )
    return Meshes(verts=[verts], faces=[faces] + [obj_f], textures=text)


def get_textured_mesh(filename, device="cuda", texture_atlas_size=1):
    verts, faces, aux = load_obj(
        filename,
        device=device,
        load_textures=True,
        create_texture_atlas=True,
        texture_atlas_size=texture_atlas_size,
        texture_wrap="repeat",
    )
    atlas = aux.texture_atlas

    return Meshes(
        verts=[verts],
        faces=[faces.verts_idx],
        textures=TexturesAtlas(atlas=[atlas]),
    )
