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


def get_renderer(
    R, T, pt_light_position=[0.0, 0.0, -1.0], image_size=256, device="cuda"
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

    return renderer


def store_images(images, image_dir):
    for i in range(len(images)):
        img_array = (images[i, ..., :3].detach().cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_array)

        img.save((image_dir / f"view_{i}.png").as_posix())


def render(
    meshes,
    R,
    T,
    pt_light_position=[0.0, 0.0, -1.0],
    image_dir="./data",
    device="cuda",
):
    renderer = get_renderer(R, T, pt_light_position, device=device)

    clone_meshes = meshes.extend(len(R))

    images = renderer(clone_meshes)

    store_images(images, image_dir)
