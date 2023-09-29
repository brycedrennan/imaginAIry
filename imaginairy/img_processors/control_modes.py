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

    depth_pt = model(img)[0]
    return depth_pt


def create_normal_map(img):
    import torch

    from imaginairy.vendored.imaginairy_normal_map.model import (
        create_normal_map_torch_img,
    )

    normal_img_t = create_normal_map_torch_img(img)
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


def make_noise_disk(H, W, C, F):
    import cv2
    import numpy as np

    noise = np.random.uniform(low=0, high=1, size=((H // F) + 2, (W // F) + 2, C))
    noise = cv2.resize(noise, (W + 2 * F, H + 2 * F), interpolation=cv2.INTER_CUBIC)
    noise = noise[F : F + H, F : F + W]
    noise -= np.min(noise)
    noise /= np.max(noise)
    if C == 1:
        noise = noise[:, :, None]
    return noise


def shuffle_map_np(img, h=None, w=None, f=256):
    import cv2
    import numpy as np

    H, W, C = img.shape
    if h is None:
        h = H
    if w is None:
        w = W

    x = make_noise_disk(h, w, 1, f) * float(W - 1)
    y = make_noise_disk(h, w, 1, f) * float(H - 1)
    flow = np.concatenate([x, y], axis=2).astype(np.float32)
    return cv2.remap(img, flow, None, cv2.INTER_LINEAR)


def shuffle_map_torch(tensor, h=None, w=None, f=256):
    import torch

    # Assuming the input tensor is in shape (B, C, H, W)
    B, C, H, W = tensor.shape
    device = tensor.device
    tensor = tensor.cpu()

    # Create an empty tensor with the same shape as input tensor to store the shuffled images
    shuffled_tensor = torch.empty_like(tensor)

    # Iterate over the batch and apply the shuffle_map function to each image
    for b in range(B):
        # Convert the input torch tensor to a numpy array
        img_np = tensor[b].numpy().transpose(1, 2, 0)  # Shape (H, W, C)

        # Call the shuffle_map function with the numpy array as input
        shuffled_np = shuffle_map_np(img_np, h, w, f)

        # Convert the shuffled numpy array back to a torch tensor and store it in the shuffled_tensor
        shuffled_tensor[b] = torch.from_numpy(
            shuffled_np.transpose(2, 0, 1)
        )  # Shape (C, H, W)
    shuffled_tensor = (shuffled_tensor + 1.0) / 2.0
    return shuffled_tensor.to(device)


def inpaint_prep(mask_image_t, target_image_t):
    """
    Combines the masked image and target image into a single tensor.

    The output tensor has any masked areas set to -1 and other pixel values set between 0 and 1.

    mask_image_t is a 3-channel torch tensor of shape (B, C, H, W) with pixel values in range [-1, 1], where -1 indicates masked areas
    target_image_t is a 3-channel torch tensor of shape (B, C, H, W) with pixel values in range [-1, 1]
    """
    import torch

    # Normalize target_image_t from [-1,1] to [0,1]
    target_image_t = (target_image_t + 1.0) / 2.0

    # Use mask_image_t to replace masked areas in target_image_t with -1
    output_image_t = torch.where(mask_image_t == -1, mask_image_t, target_image_t)

    return output_image_t


def to_grayscale(img):
    # The dimensions of input should be (batch_size, channels, height, width)
    if img.dim() != 4:
        raise ValueError("Input should be a 4d tensor")
    if img.size(1) != 3:
        raise ValueError("Input should have 3 channels")

    # Apply the formula to convert to grayscale.
    gray = (
        0.2989 * img[:, 0, :, :] + 0.5870 * img[:, 1, :, :] + 0.1140 * img[:, 2, :, :]
    )

    # Expand the dimensions so it's a 1-channel image.
    gray = gray.unsqueeze(1)

    # Duplicate the single channel to have 3 identical channels
    gray_3_channels = gray.repeat(1, 3, 1, 1)

    return (gray_3_channels + 1.0) / 2.0


def noop(img):
    return (img + 1.0) / 2.0


CONTROL_MODES = {
    "canny": create_canny_edges,
    "depth": create_depth_map,
    "normal": create_normal_map,
    "hed": create_hed_edges,
    # "mlsd": create_mlsd_edges,
    "openpose": create_pose_map,
    # "scribble": None,
    "shuffle": shuffle_map_torch,
    "edit": noop,
    "inpaint": inpaint_prep,
    "details": noop,
    "colorize": to_grayscale,
}
