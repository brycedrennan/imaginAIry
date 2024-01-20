import os

import pytest
from PIL import Image

from imaginairy.enhancers.upscalers.realesrgan import upscale_image
from tests import TESTS_FOLDER
from tests.utils import assert_image_similar_to_expectation

upscale_challenges_folder = f"{TESTS_FOLDER}/data/upscale_challenges"
def test_upscale_textured_image(filename_base_for_outputs):
    img = Image.open(f"{upscale_challenges_folder}/sand.jpg")
    upscaled_image = upscale_image(img, ultrasharp=True)
    assert_image_similar_to_expectation(
        upscaled_image, f"{filename_base_for_outputs}.jpg", threshold=25000
    )


@pytest.mark.skip()
def test_upscalers_difficult_images(filename_base_for_outputs):
    weight_urls = [
        # "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",  # blurry on sand
        "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/1856559b50de25116a7c07261177dd128f1f5664/4x-UltraSharp.pth",
        "https://github.com/Phhofm/models/raw/main/4xLSDIRplus/4xLSDIRplusC.pth",
        "https://github.com/Phhofm/models/raw/main/4xLSDIRplus/4xLSDIRplusN.pth",
        # "https://github.com/Phhofm/models/raw/main/4xLSDIRplus/4xLSDIRplusR.pth",  # blurry on sand
        "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_RealisticRescaler_100000_G.pth",
        "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/UniversalUpscaler/4x_UniversalUpscalerV2-Neutral_115000_swaG.pth",
        "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/UniversalUpscaler/4x_UniversalUpscalerV2-Sharper_103000_G.pth",
        "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/UniversalUpscaler/4x_UniversalUpscalerV2-Sharp_101000_G.pth",
        "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_foolhardy_Remacri.pth",
        "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_Valar_v1.pth",
        "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_NMKDSuperscale_Artisoft_120000_G.pth",
        "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_NMKD-Superscale-SP_178000_G.pth",
        "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4xPSNR.pth",
    ]
    for img_filename in os.listdir(upscale_challenges_folder):
        if not img_filename.endswith(".jpg"):
            continue
        img_name = img_filename.split(".")[0]
        img = Image.open(f"{upscale_challenges_folder}/{img_filename}")
        for url in weight_urls:
            weights_filename = url.split("/")[-1]
            upscaled_image = upscale_image(img, weights_url=url)
            upscaled_image.save(
                f"{filename_base_for_outputs}_{img_name}_{weights_filename}.jpg"
            )
