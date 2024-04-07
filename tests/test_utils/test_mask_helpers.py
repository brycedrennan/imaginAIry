import torch
from PIL import Image, ImageOps

from imaginairy.api import imagine
from imaginairy.enhancers.clip_masking import get_img_mask
from imaginairy.schema import ImaginePrompt
from imaginairy.utils import img_convert
from imaginairy.utils.img_convert import (
    pillow_img_to_torch_image,
    pillow_mask_255_to_torch_mask,
    torch_img_to_pillow_img,
)
from imaginairy.utils.img_utils import blur_fill, combine_image
from imaginairy.utils.mask_helpers import (
    fill_navier_stokes,
    fill_neutral,
    fill_noise,
    highlight_masked_area,
)
from tests import TESTS_FOLDER


def makemask(mask, offset=0.1, threshold=0.2):
    B, C, H, W = mask.shape
    if C == 3:
        mask = mask.mean(dim=1, keepdim=True)

    assert 0.0 <= offset < threshold <= 1.0, "Threshold must be higher than offset"
    mask = (mask - offset) * (1 / (threshold - offset))
    mask = mask.clamp(0, 1)
    return mask


def test_fill_neutral(filename_base_for_outputs):
    img = Image.open(f"{TESTS_FOLDER}/data/bench2.png").convert("RGB")
    mask = Image.open(f"{TESTS_FOLDER}/data/bench2_mask.png")

    img_t = pillow_img_to_torch_image(img)
    mask_t = pillow_img_to_torch_image(mask)
    mask_t = makemask(mask_t)
    for falloff in [0, 1, 3, 5, 17]:
        filled_img_t = fill_neutral(img_t, mask_t, falloff=falloff)
        filled_img = torch_img_to_pillow_img(filled_img_t)
        img_path = f"{filename_base_for_outputs}_filled_neutral_falloff_{falloff}.png"
        filled_img.save(img_path)
        # assert_image_similar_to_expectation(filled_img, img_path=img_path, threshold=7000)


def test_fill_navier_stokes(filename_base_for_outputs):
    img = Image.open(f"{TESTS_FOLDER}/data/bench2.png").convert("RGB")
    mask = Image.open(f"{TESTS_FOLDER}/data/bench2_mask.png")

    img_t = pillow_img_to_torch_image(img)
    mask_t = pillow_img_to_torch_image(mask)
    mask_t = makemask(mask_t)
    for falloff in [0, 1, 3, 5, 17]:
        filled_img_t = fill_navier_stokes(img_t, mask_t, falloff=falloff)
        filled_img = torch_img_to_pillow_img(filled_img_t)
        img_path = f"{filename_base_for_outputs}_filled_neutral_falloff_{falloff}.png"
        filled_img.save(img_path)


def test_inpaint_prep_dogbench(filename_base_for_outputs):
    save_count = 0

    def save(i, name):
        nonlocal save_count
        if isinstance(i, torch.Tensor):
            i = torch_img_to_pillow_img(i)
        i.save(f"{filename_base_for_outputs}_{save_count:02d}_{name}.png")
        save_count += 1

    img = Image.open(f"{TESTS_FOLDER}/data/dog-on-bench.png").convert("RGB")
    img_t = pillow_img_to_torch_image(img)
    save(img, "original")

    mask_img, mask_img_g = get_img_mask(img, "dog", threshold=0.5)
    save(mask_img_g, "mask_g")
    mask_img_g_t = pillow_mask_255_to_torch_mask(mask_img_g)
    print(
        f"mask_img_g value range: {mask_img_g_t.min().item()} - {mask_img_g_t.max().item()}"
    )
    mask_highlight_g_t = highlight_masked_area(
        img_t, mask_img_g_t, color=(255, 0, 0), highlight_strength=1
    )
    save(mask_highlight_g_t, "highlighted-mask_g")

    save(mask_img, "mask")

    mask_t = pillow_mask_255_to_torch_mask(mask_img)
    mask_t = makemask(mask_t)

    mask_highlight_t = highlight_masked_area(
        img_t,
        mask_t,
        #  color=(255, 0, 0)
    )
    save(mask_highlight_t, "highlighted-mask")

    navier_filled_img_t = fill_navier_stokes(img_t, mask_t, falloff=0)
    save(navier_filled_img_t, "filled-navier-stokes")

    # blur the filled area
    blur_filled_img_t = blur_fill(navier_filled_img_t, mask=mask_t, blur=20, falloff=40)
    save(blur_filled_img_t, "navier-blurred-filled")

    # neutral fill the masked area
    neutral_filled_img_t = fill_neutral(img_t, mask_t, falloff=1)
    save(neutral_filled_img_t, "filled-neutral")

    # noise fill the masked area
    noise_filled_img_t = fill_noise(img_t, mask_t)
    save(noise_filled_img_t, "filled-noise")

    seed = 2
    prompt_text = "a red fox on a bench"

    prompts = [
        ImaginePrompt(
            prompt_text,
            init_image=img,
            init_image_strength=0,
            mask_image=mask_img,
            seed=seed,
            model_weights="sdxl",
            caption_text="original-filled",
        ),
        ImaginePrompt(
            prompt_text,
            init_image=img_convert.torch_img_to_pillow_img(neutral_filled_img_t),
            init_image_strength=0,
            mask_image=mask_img,
            seed=seed,
            model_weights="sdxl",
            caption_text="neutral-filled",
        ),
        ImaginePrompt(
            prompt_text,
            init_image=img_convert.torch_img_to_pillow_img(noise_filled_img_t),
            init_image_strength=0,
            mask_image=mask_img,
            seed=seed,
            model_weights="sdxl",
            caption_text="noise-filled",
        ),
        ImaginePrompt(
            prompt_text,
            init_image=img_convert.torch_img_to_pillow_img(blur_filled_img_t),
            init_image_strength=0,
            mask_image=mask_img,
            seed=seed,
            model_weights="sdxl",
            caption_text="navier-stokes-filled",
        ),
    ]

    for result in imagine(prompts):
        generated_img = result.images["pre-reconstitution"]
        save(generated_img, f"{result.prompt.caption_text}_pre-reconstitution")

        rebuilt_img = combine_image(
            original_img=img,
            generated_img=generated_img,
            mask_img=ImageOps.invert(mask_img),
        )
        save(rebuilt_img, f"{result.prompt.caption_text}_rebuilt")

        # for img_name, img in result.images.items():
        #     if "mask" in img_name:
        #         continue
        #
        #     name = f"{result.prompt.caption_text}_{img_name}"
        #     save(img, name)
