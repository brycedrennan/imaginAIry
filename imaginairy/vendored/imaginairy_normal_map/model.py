from functools import lru_cache

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import nn
from torchvision import transforms

from .decoder import Decoder
from .encoder import Encoder
from .utils import get_device


class NNET(nn.Module):
    def __init__(self, architecture="BN", sampling_ratio=0.4, importance_ratio=0.7):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(
            architecture=architecture,
            sampling_ratio=sampling_ratio,
            importance_ratio=importance_ratio,
        )

    def forward(self, img, **kwargs):
        return self.decoder(self.encoder(img), **kwargs)


def create_normal_map_pil_img(img, device=get_device()):
    img_t = pillow_img_to_torch_normal_map_input(img).to(device)
    pred_norm = create_normal_map_torch_img(img_t, device=device)
    return torch_normal_map_to_pillow_img(pred_norm)


def create_normal_map_torch_img(img_t, device=get_device()):
    with torch.no_grad():
        model = load_model(device=device)
        img_t = img_t.to(device)
        norm_out_list, _, _ = model(img_t)  # noqa
        norm_out = norm_out_list[-1]

        pred_norm_t = norm_out[:, :3, :, :]
        return pred_norm_t


normalize_img = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


def pillow_img_to_torch_normal_map_input(img):
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = normalize_img(img)
    img = img.unsqueeze(0)

    # Resize image to nearest multiple of 8 using interpolate()
    h, w = img.size(2), img.size(3)
    h_new = int(round(h / 8) * 8)
    w_new = int(round(w / 8) * 8)
    img = torch.nn.functional.interpolate(
        img, size=(h_new, w_new), mode="bilinear", align_corners=False
    )

    return img


def torch_normal_map_to_pillow_img(norm_map_t):
    norm_map_np = norm_map_t.detach().cpu().permute(0, 2, 3, 1).numpy()  # (B, H, W, 3)
    pred_norm_rgb = ((norm_map_np + 1) * 0.5) * 255
    pred_norm_rgb = np.clip(pred_norm_rgb, a_min=0, a_max=255)
    pred_norm_rgb = pred_norm_rgb.astype(np.uint8)  # (B, H, W, 3)
    return Image.fromarray(pred_norm_rgb[0])


def load_checkpoint(fpath, model):
    ckpt = torch.load(fpath, map_location="cpu")["model"]

    load_dict = {}
    for k, v in ckpt.items():
        load_dict[k] = v

    model.load_state_dict(load_dict)
    return model


@lru_cache(maxsize=1)
def load_model(device=None, sampling_ratio=0.4, importance_ratio=0.7) -> NNET:
    device = device or get_device()
    weights_path = hf_hub_download(
        repo_id="imaginairy/imaginairy-normal-uncertainty-map", filename="scannet.pt"
    )
    architecture = "BN"

    model = NNET(
        architecture=architecture,
        sampling_ratio=sampling_ratio,
        importance_ratio=importance_ratio,
    ).to(device)
    model = load_checkpoint(weights_path, model)
    model.eval()
    return model
