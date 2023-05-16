import numpy as np
import torch
from PIL import Image

from imaginairy.model_manager import get_cached_url_path
from imaginairy.utils import get_device
from imaginairy.utils.model_cache import memory_managed_model
from imaginairy.vendored.basicsr.rrdbnet_arch import RRDBNet
from imaginairy.vendored.realesrgan import RealESRGANer


@memory_managed_model("realesrgan_upsampler", memory_usage_mb=70)
def realesrgan_upsampler():
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
    )
    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    model_path = get_cached_url_path(url)
    device = get_device()

    upsampler = RealESRGANer(
        scale=4, model_path=model_path, model=model, tile=512, device=device
    )

    upsampler.device = torch.device(device)
    upsampler.model.to(device)

    return upsampler


def upscale_image(img):
    img = img.convert("RGB")

    np_img = np.array(img, dtype=np.uint8)
    upsampler_output, img_mode = realesrgan_upsampler().enhance(np_img[:, :, ::-1])
    return Image.fromarray(upsampler_output[:, :, ::-1], mode=img_mode)
