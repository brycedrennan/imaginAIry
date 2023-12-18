"""Code for depth estimation with MiDaS models"""

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
    "dpt_beit_large_512": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt",
    "dpt_beit_large_384": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_384.pt",
    "dpt_beit_base_384": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_base_384.pt",
    # "dpt_swin2_large_384": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_large_384.pt",
    # "dpt_swin2_base_384": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_base_384.pt",
    # "dpt_swin2_tiny_256": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_tiny_256.pt",
    # "dpt_swin_large_384": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin_large_384.pt",
    # "dpt_next_vit_large_384": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_next_vit_large_384.pt",
    # "dpt_levit_224": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_levit_224.pt",
    "dpt_large_384": "https://github.com/isl-org/MiDaS/releases/download/v3/dpt_large_384.pt",
    "dpt_hybrid_384": "https://github.com/isl-org/MiDaS/releases/download/v3/dpt_hybrid_384.pt",
    "midas_v21_384": "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_384.pt",
    # "midas_v21_small_256": "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt",
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
    if model_type in ("dpt_large_384", "dpt_large_384_v1"):  # DPT-Large
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type in ("dpt_hybrid_384", "dpt_hybrid_384_v1"):  # DPT-Hybrid
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
def load_model(
    # device,
    # model_path,
    model_type="dpt_large_384",
    optimize=True,
    height=None,
    square=False,
):
    """Load the specified network.

    Args:
        device (device): the torch device used
        model_path (str): path to saved model
        model_type (str): the type of the model to be loaded
        optimize (bool): optimize the model to half-integer on CUDA?
        height (int): inference encoder image height
        square (bool): resize to a square resolution?

    Returns:
        The loaded network, the transform which prepares images as input to the network and the dimensions of the
        network input
    """
    model_path = ISL_PATHS[model_type]

    keep_aspect_ratio = not square

    if model_type == "dpt_beit_large_512":
        model = DPTDepthModel(
            path=model_path,
            backbone="beitl16_512",
            non_negative=True,
        )
        net_w, net_h = 512, 512
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_beit_large_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="beitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_beit_base_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="beitb16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_swin2_large_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="swin2l24_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_swin2_base_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="swin2b24_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_swin2_tiny_256":
        model = DPTDepthModel(
            path=model_path,
            backbone="swin2t16_256",
            non_negative=True,
        )
        net_w, net_h = 256, 256
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_swin_large_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="swinl12_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_next_vit_large_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="next_vit_large_6m",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    # We change the notation from dpt_levit_224 (MiDaS notation) to levit_384 (timm notation) here, where the 224 refers
    # to the resolution 224x224 used by LeViT and 384 is the first entry of the embed_dim, see _cfg and model_cfgs of
    # https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/levit.py
    # (commit id: 927f031293a30afb940fff0bee34b85d9c059b0e)
    elif model_type == "dpt_levit_224":
        model = DPTDepthModel(
            path=model_path,
            backbone="levit_384",
            non_negative=True,
            head_features_1=64,
            head_features_2=8,
        )
        net_w, net_h = 224, 224
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type in ("dpt_large_384", "dpt_large_384_v1"):
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type in ("dpt_hybrid_384", "dpt_hybrid_384_v1"):
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "midas_v21_384":
        model = MidasNet(model_path, non_negative=True)
        net_w, net_h = 384, 384
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    elif model_type == "midas_v21_small_256":
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

    if "openvino" not in model_type:
        print(
            "Model loaded, number of parameters = {:.0f}M".format(
                sum(p.numel() for p in model.parameters()) / 1e6
            )
        )
    else:
        print("Model loaded, optimized with OpenVINO")

    if "openvino" in model_type:
        keep_aspect_ratio = False

    if height is not None:
        net_w, net_h = height, height

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=keep_aspect_ratio,
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
    def __init__(self, model_type):
        super().__init__()
        # assert model_type in self.MODEL_TYPES_ISL
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
