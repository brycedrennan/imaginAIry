"""Functions to create hint images for controlnet."""


def create_canny_edges(img):
    import cv2
    import numpy as np
    import torch
    from einops import einops

    img = torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0)
    img = einops.rearrange(img[0], "c h w -> h w c")
    img = (255.0 * img).cpu().numpy().astype(np.uint8).squeeze()
    blurred = cv2.GaussianBlur(img, (5, 5), 0).astype(np.uint8)

    if len(blurred.shape) > 2:
        blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    threshold2, _ = cv2.threshold(
        blurred, thresh=0, maxval=255, type=(cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    )
    canny_image = cv2.Canny(
        blurred, threshold1=(threshold2 * 0.5), threshold2=threshold2
    )

    # canny_image = cv2.Canny(blur, 100, 200)
    canny_image = canny_image[:, :, None]
    # controlnet requires three channels
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    canny_image = torch.from_numpy(canny_image).to(dtype=torch.float32) / 255.0
    canny_image = einops.rearrange(canny_image, "h w c -> c h w").clone()
    canny_image = canny_image.unsqueeze(0)

    return canny_image


def create_depth_map(img):
    import torch

    orig_size = img.shape[2:]

    depth_pt = _create_depth_map_raw(img)
    # copy the depth map to the other channels
    depth_pt = torch.cat([depth_pt, depth_pt, depth_pt], dim=0)

    depth_pt -= torch.min(depth_pt)
    depth_pt /= torch.max(depth_pt)
    depth_pt = depth_pt.unsqueeze(0)
    # depth_pt = depth_pt.cpu().numpy()

    depth_pt = torch.nn.functional.interpolate(
        depth_pt,
        size=orig_size,
        mode="bilinear",
    )

    return depth_pt


def _create_depth_map_raw(img):
    import torch

    from imaginairy.modules.midas.api import MiDaSInference, midas_device

    model = MiDaSInference(model_type="dpt_hybrid").to(midas_device())
    img = img.to(midas_device())
    max_size = 512

    # calculate new size such that image fits within 512x512 but keeps aspect ratio
    if img.shape[2] > img.shape[3]:
        new_size = (max_size, int(max_size * img.shape[3] / img.shape[2]))
    else:
        new_size = (int(max_size * img.shape[2] / img.shape[3]), max_size)

    # resize torch image to be multiple of 32
    img = torch.nn.functional.interpolate(
        img,
        size=(new_size[0] // 32 * 32, new_size[1] // 32 * 32),
        mode="bilinear",
        align_corners=False,
    )

    depth_pt = model(img)[0]  # noqa
    return depth_pt


def create_normal_map_old(img):
    import cv2
    import numpy as np
    import torch

    depth = _create_depth_map_raw(img)
    depth = depth[0]

    depth_pt = depth.clone()
    depth_pt -= torch.min(depth_pt)
    depth_pt /= torch.max(depth_pt)
    depth_pt = depth_pt.cpu().numpy()

    bg_th = 0.1
    a = np.pi * 2.0
    depth_np = depth.cpu().float().numpy()
    x = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=3)
    y = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=3)
    z = np.ones_like(x) * a
    x[depth_pt < bg_th] = 0
    y[depth_pt < bg_th] = 0
    normal = np.stack([x, y, z], axis=2)
    normal /= np.sum(normal**2.0, axis=2, keepdims=True) ** 0.5
    normal_image = (normal * 127.5 + 127.5).clip(0, 255).astype(np.uint8)

    normal_image = torch.from_numpy(normal_image[:, :, ::-1].copy()).float() / 255.0
    normal_image = normal_image.permute(2, 0, 1)
    normal_image = normal_image.unsqueeze(0)
    # for use with Controlnet 1.1?
    # normal_image = normal_image[:, [1, 0, 2], :, :]

    return normal_image


def create_normal_map(img):
    import torch
    from imaginairy_normal_map.model import create_normal_map_torch_img

    normal_img_t = create_normal_map_torch_img(img)
    # normal_img_t = normal_img_t[:, [1, 2, 0], :, :]
    normal_img_t -= torch.min(normal_img_t)
    normal_img_t /= torch.max(normal_img_t)

    return normal_img_t


def create_hed_edges(img_t):
    import torch

    from imaginairy.img_processors.hed_boundary import create_hed_map
    from imaginairy.utils import get_device

    img_t = img_t.to(get_device())
    # rgb to bgr
    img_t = img_t[:, [2, 1, 0], :, :]

    hint_t = create_hed_map(img_t)
    hint_t = hint_t.unsqueeze(0)
    hint_t = torch.cat([hint_t, hint_t, hint_t], dim=0)

    hint_t -= torch.min(hint_t)
    hint_t /= torch.max(hint_t)
    hint_t = (hint_t * 255).clip(0, 255).to(dtype=torch.uint8).float() / 255.0

    hint_t = hint_t.unsqueeze(0)
    # hint_t = hint_t[:, [2, 0, 1], :, :]
    return hint_t


def create_pose_map(img_t):
    from imaginairy.img_processors.openpose import create_body_pose_img
    from imaginairy.utils import get_device

    img_t = img_t.to(get_device())
    pose_t = create_body_pose_img(img_t) / 255
    # pose_t = pose_t[:, [2, 1, 0], :, :]
    return pose_t


CONTROL_MODES = {
    "canny": create_canny_edges,
    "depth": create_depth_map,
    "normal": create_normal_map,
    "hed": create_hed_edges,
    # "mlsd": create_mlsd_edges,
    "openpose": create_pose_map,
    # "scribble": None,
}
