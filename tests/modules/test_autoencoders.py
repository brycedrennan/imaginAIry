import numpy as np
import pytest
from PIL import Image
from torch.nn.functional import interpolate

from imaginairy.enhancers.upscale_riverwing import upscale_latent
from imaginairy.schema import LazyLoadingImage
from imaginairy.utils import get_device
from imaginairy.utils.img_utils import (
    pillow_fit_image_within,
    pillow_img_to_torch_image,
    torch_img_to_pillow_img,
)
from imaginairy.utils.model_manager import get_diffusion_model
from tests import TESTS_FOLDER

strat_combos = [
    ("sliced", "sliced"),
    ("sliced", "all_at_once"),
    ("folds", "folds"),
    ("folds", "all_at_once"),
    ("all_at_once", "all_at_once"),
    ("all_at_once", "sliced"),
    ("all_at_once", "folds"),
]


@pytest.mark.skipif(True, reason="Run manually as needed. Uses too much memory.")
@pytest.mark.parametrize(("encode_strat", "decode_strat"), strat_combos)
def test_encode_decode(filename_base_for_outputs, encode_strat, decode_strat):
    """
    Test that encoding and decoding works.

    Outputs comparison with original image.
    """
    model = get_diffusion_model()
    img = LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/beach_at_sainte_adresse.jpg")
    img = pillow_fit_image_within(img, max_height=img.height, max_width=img.width)
    img.save(f"{filename_base_for_outputs}_orig.png")
    img_t = pillow_img_to_torch_image(img).to(get_device())
    if encode_strat == "all_at_once":
        latent = model.first_stage_model.encode_all_at_once(img_t) * model.scale_factor
    elif encode_strat == "folds":
        latent = model.first_stage_model.encode_with_folds(img_t) * model.scale_factor
    else:
        latent = model.first_stage_model.encode_sliced(img_t) * model.scale_factor

    if decode_strat == "all_at_once":
        decoded_img_t = model.first_stage_model.decode_all_at_once(
            latent / model.scale_factor
        )
    elif decode_strat == "folds":
        decoded_img_t = model.first_stage_model.decode_with_folds(
            latent / model.scale_factor
        )
    else:
        decoded_img_t = model.first_stage_model.decode_sliced(
            latent / model.scale_factor
        )
    decoded_img_t = interpolate(decoded_img_t, img_t.shape[-2:])
    decoded_img = torch_img_to_pillow_img(decoded_img_t)
    decoded_img.save(f"{filename_base_for_outputs}.png")
    diff_img = Image.fromarray(np.asarray(img) - np.asarray(decoded_img))
    diff_img.save(f"{filename_base_for_outputs}_diff.png")


@pytest.mark.skip()
def test_encode_decode_naive_scale(filename_base_for_outputs):
    model = get_diffusion_model()
    img = LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/dog.jpg")
    img = pillow_fit_image_within(img, max_height=img.height, max_width=img.width)
    img.save(f"{filename_base_for_outputs}_orig.png")
    img_t = pillow_img_to_torch_image(img).to(get_device())
    latent = model.first_stage_model.encode_sliced(img_t) * model.scale_factor
    latent = interpolate(latent, scale_factor=2)

    decoded_img_t = model.first_stage_model.decode_sliced(latent / model.scale_factor)
    decoded_img = torch_img_to_pillow_img(decoded_img_t)
    decoded_img.save(f"{filename_base_for_outputs}.png")


@pytest.mark.skip(reason="experimental")
def test_upscale_methods(filename_base_for_outputs, steps):
    """
    compare upscale methods.
    """
    steps = 25
    model = get_diffusion_model()
    roi_pcts = (0.7, 0.1, 0.9, 0.3)

    def crop_pct(img, roi_pcts):
        w, h = img.size
        roi = (
            int(w * roi_pcts[0]),
            int(h * roi_pcts[1]),
            int(w * roi_pcts[2]),
            int(h * roi_pcts[3]),
        )
        return img.crop(roi)

    def decode(latent):
        t = model.first_stage_model.decode_sliced(latent / model.scale_factor)
        return torch_img_to_pillow_img(t)

    img = LazyLoadingImage(
        filepath=f"{TESTS_FOLDER}/data/010853_1_kdpmpp2m30_PS7.5_portrait_photo_of_a_freckled_woman_[generated].jpg"
    )
    img = pillow_fit_image_within(img, max_height=img.height, max_width=img.width)
    img = crop_pct(img, roi_pcts)

    upscaled = []
    sampling_methods = [
        ("nearest", Image.Resampling.NEAREST),
        ("bilinear", Image.Resampling.BILINEAR),
        ("bicubic", Image.Resampling.BICUBIC),
        ("lanczos", Image.Resampling.LANCZOS),
    ]
    for method_name, sample_method in sampling_methods:
        upscaled.append(
            (
                img.resize((img.width * 4, img.height * 4), resample=sample_method),
                f"{method_name}",
            )
        )

    img_t = pillow_img_to_torch_image(img).to(get_device())
    latent = model.first_stage_model.encode_sliced(img_t) * model.scale_factor

    sharp_latent = upscale_latent(
        latent, steps=steps, upscale_prompt="high detail, sharp focus, 4k"
    )
    sharp_latent = upscale_latent(
        sharp_latent, steps=steps, upscale_prompt="high detail, sharp focus, 4k"
    )
    upscaled.append((decode(sharp_latent), "riverwing-upscaler-sharp"))

    blurry_latent = upscale_latent(
        latent, steps=steps, upscale_prompt="blurry, low detail, 360p"
    )
    blurry_latent = upscale_latent(
        blurry_latent, steps=steps, upscale_prompt="blurry, low detail, 360p"
    )
    upscaled.append((decode(blurry_latent), "riverwing-upscaler-blurry"))

    # upscaled.append((decode(latent).resize(), "original"))

    for img, name in upscaled:
        img.resize((img.width, img.height), resample=Image.NEAREST).save(
            f"{filename_base_for_outputs}_{name}.jpg"
        )
