import numpy as np
from PIL import Image
from pytorch3d.renderer import (
    BlendParams,
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
)


def render(
    meshes,
    R,
    T,
    pt_light_position=[0.0, 0.0, -1.0],
    image_size=1024,
    image_dir="./data",
    device="cuda",
):
    cameras = FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device,
    )
    pt_lights = PointLights(device=device, location=[pt_light_position])

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0,
        faces_per_pixel=1,
        bin_size=49,
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=pt_lights,
            blend_params=BlendParams(background_color=[0.8] * 3),
        ),
    )

    clone_meshes = meshes.extend(len(cameras))

    images = renderer(clone_meshes, cameras=cameras, lights=pt_lights)

    for i in range(len(cameras)):
        img_array = (images[i, ..., :3].cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_array)

        img.save((image_dir / f"view_{i}.png").as_posix())
