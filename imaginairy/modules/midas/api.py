# based on https://github.com/isl-org/MiDaS
from functools import lru_cache

import cv2
import torch
from einops import rearrange
from torch import nn
from torchvision.transforms import Compose

from imaginairy.modules.midas.midas.dpt_depth import DPTDepthModel
from imaginairy.modules.midas.midas.midas_net import MidasNet
from imaginairy.modules.midas.midas.midas_net_custom import MidasNet_small
from imaginairy.modules.midas.midas.transforms import (
    NormalizeImage,
    PrepareForNet,
    Resize,
)
from imaginairy.utils import get_device

ISL_PATHS = {
    "dpt_large": "midas_models/dpt_large-midas-2f21e586.pt",
    "dpt_hybrid": "midas_models/dpt_hybrid-midas-501f0c75.pt",
    "midas_v21": "",
    "midas_v21_small": "",
}


def disabled_train(self, mode=True):
    """
    Overwrite model.train with this function to make sure train/eval mode
    does not change anymore.
    """
    return self


def load_midas_transform(model_type="dpt_hybrid"):
    # https://github.com/isl-org/MiDaS/blob/master/run.py
    # load transform only
    if model_type == "dpt_large":  # DPT-Large
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_hybrid":  # DPT-Hybrid
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "midas_v21":
        net_w, net_h = 384, 384
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    elif model_type == "midas_v21_small":
        net_w, net_h = 256, 256
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    else:
        msg = f"model_type '{model_type}' not implemented, use: --model_type large"
        raise NotImplementedError(msg)

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    return transform


@lru_cache(maxsize=1)
def load_model(model_type):
    # https://github.com/isl-org/MiDaS/blob/master/run.py
    # load network
    model_path = ISL_PATHS[model_type]
    if model_type == "dpt_large":  # DPT-Large
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_hybrid":  # DPT-Hybrid
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "midas_v21":
        model = MidasNet(model_path, non_negative=True)
        net_w, net_h = 384, 384
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    elif model_type == "midas_v21_small":
        model = MidasNet_small(
            model_path,
            features=64,
            backbone="efficientnet_lite3",
            exportable=True,
            non_negative=True,
            blocks={"expand": True},
        )
        net_w, net_h = 256, 256
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    else:
        msg = f"model_type '{model_type}' not implemented, use: --model_type large"
        raise NotImplementedError(msg)

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    return model.eval(), transform


@lru_cache
def midas_device():
    # mps returns incorrect results ~50% of the time
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache
def load_midas(model_type="dpt_hybrid"):
    model = MiDaSInference(model_type)
    model.to(midas_device())
    return model


def torch_image_to_depth_map(image_t: torch.Tensor, model_type="dpt_hybrid"):
    model = load_midas(model_type)
    transform = load_midas_transform(model_type)
    image_t = rearrange(image_t, "b c h w -> b h w c")[0]
    image_np = ((image_t + 1.0) * 0.5).detach().cpu().numpy()
    image_np = transform({"image": image_np})["image"]
    image_t = torch.from_numpy(image_np[None, ...])
    image_t = image_t.to(device=midas_device())

    depth_t = model(image_t)
    depth_min = torch.amin(depth_t, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_t, dim=[1, 2, 3], keepdim=True)

    depth_t = (depth_t - depth_min) / (depth_max - depth_min)
    return depth_t.to(get_device())


class MiDaSInference(nn.Module):
    MODEL_TYPES_TORCH_HUB = ["DPT_Large", "DPT_Hybrid", "MiDaS_small"]
    MODEL_TYPES_ISL = [
        "dpt_large",
        "dpt_hybrid",
        "midas_v21",
        "midas_v21_small",
    ]

    def __init__(self, model_type):
        super().__init__()
        assert model_type in self.MODEL_TYPES_ISL
        model, _ = load_model(model_type)
        self.model = model
        self.model.train = disabled_train
        self.model.eval()

    def forward(self, x):
        # x in 0..1 as produced by calling self.transform on a 0..1 float64 numpy array
        # NOTE: we expect that the correct transform has been called during dataloading.
        with torch.no_grad():
            prediction = self.model(x)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=x.shape[2:],
                mode="bicubic",
                align_corners=False,
            )
        assert prediction.shape == (x.shape[0], 1, x.shape[2], x.shape[3])
        return prediction
