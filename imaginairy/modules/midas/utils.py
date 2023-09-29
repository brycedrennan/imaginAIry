"""Utils for monoDepth."""
import re
import sys

import cv2
import numpy as np
import torch

from imaginairy.modules.midas.api import load_midas_transform


class AddMiDaS:
    def __init__(self, model_type="dpt_hybrid"):
        self.transform = load_midas_transform(model_type)

    def pt2np(self, x):
        x = ((x + 1.0) * 0.5).detach().cpu().numpy()
        return x

    def np2pt(self, x):
        x = torch.from_numpy(x) * 2 - 1.0
        return x

    def __call__(self, img):
        # sample['jpg'] is tensor hwc in [-1, 1] at this point
        img = self.pt2np(img)
        img = self.transform({"image": img})["image"]
        return img


def read_pfm(path):
    """
    Read pfm file.

    Args:
        path (str): path to file

    Returns:
        tuple: (data, scale)
    """
    with open(path, "rb") as file:
        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise ValueError("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise RuntimeError("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale


def write_pfm(path, image, scale=1):
    """
    Write pfm file.

    Args:
        path (str): pathto file
        image (array): data
        scale (int, optional): Scale. Defaults to 1.
    """

    with open(path, "wb") as file:
        if image.dtype.name != "float32":
            raise ValueError("Image dtype must be float32.")

        image = np.flipud(image)

        if len(image.shape) == 3 and image.shape[2] == 3:  # color image
            color = True
        elif (
            len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1
        ):  # greyscale
            color = False
        else:
            msg = "Image must have H x W x 3, H x W x 1 or H x W dimensions."
            raise ValueError(msg)

        file.write("PF\n" if color else b"Pf\n")
        file.write(b"%d %d\n" % (image.shape[1], image.shape[0]))

        endian = image.dtype.byteorder

        if endian == "<" or endian == "=" and sys.byteorder == "little":
            scale = -scale

        file.write(b"%f\n" % scale)

        image.tofile(file)


def read_image(path):
    """
    Read image and output RGB image (0-1).

    Args:
        path (str): path to file

    Returns:
        array: RGB image (0-1)
    """
    img = cv2.imread(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img


def resize_image(img):
    """
    Resize image and make it fit for network.

    Args:
        img (array): image

    Returns:
        tensor: data ready for network
    """
    height_orig = img.shape[0]
    width_orig = img.shape[1]

    scale = width_orig / 384 if width_orig > height_orig else height_orig / 384

    height = (np.ceil(height_orig / scale / 32) * 32).astype(int)
    width = (np.ceil(width_orig / scale / 32) * 32).astype(int)

    img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    img_resized = (
        torch.from_numpy(np.transpose(img_resized, (2, 0, 1))).contiguous().float()
    )
    img_resized = img_resized.unsqueeze(0)

    return img_resized


def resize_depth(depth, width, height):
    """
    Resize depth map and bring to CPU (numpy).

    Args:
        depth (tensor): depth
        width (int): image width
        height (int): image height

    Returns:
        array: processed depth
    """
    depth = torch.squeeze(depth[0, :, :, :]).to("cpu")

    depth_resized = cv2.resize(
        depth.numpy(), (width, height), interpolation=cv2.INTER_CUBIC
    )

    return depth_resized


def write_depth(path, depth, bits=1):
    """
    Write depth map to pfm and png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
    """
    write_pfm(path + ".pfm", depth.astype(np.float32))

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2 ** (8 * bits)) - 1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.type)

    if bits == 1:
        cv2.imwrite(path + ".png", out.astype("uint8"))
    elif bits == 2:
        cv2.imwrite(path + ".png", out.astype("uint16"))
