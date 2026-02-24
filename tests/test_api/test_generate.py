import os.path

import pytest

from imaginairy.api import imagine, imagine_image_files
from imaginairy.img_processors.control_modes import CONTROL_MODES
from imaginairy.schema import ControlInput, ImaginePrompt, LazyLoadingImage, MaskMode
from imaginairy.utils import get_device
from imaginairy.utils.img_utils import pillow_fit_image_within
from tests import TESTS_FOLDER
from tests.utils import assert_image_similar_to_expectation


def test_imagine(solver_type, filename_base_for_outputs):
    prompt_text = "a scenic old-growth forest with diffuse light poking through the canopy. high resolution nature photography"
    prompt = ImaginePrompt(
        prompt_text, size=512, steps=20, seed=1, solver_type=solver_type
    )
    result = next(imagine(prompt))

    threshold_lookup = {"k_dpm_2_a": 26000}
    threshold = threshold_lookup.get(solver_type, 10000)

    img_path = f"{filename_base_for_outputs}.png"
    assert_image_similar_to_expectation(
        result.img, img_path=img_path, threshold=threshold
    )


compare_prompts = [
    "a photo of a bowl of fruit",
    "a headshot photo of a happy couple smiling at the camera",
    "a painting of a beautiful cloudy sunset at the beach",
    "a photo of a dog",
    "a photo of a handshake",
    "a photo of an astronaut riding a horse on the moon. the earth visible in the background",
]


@pytest.mark.skipif(get_device() != "cuda", reason="Too slow to run on CPU or MPS")
@pytest.mark.parametrize("model_version", ["SD-1.5"])
def test_model_versions(filename_base_for_orig_outputs, model_version):
    """Test that we can switch between model versions."""
    prompts = []
    for prompt_text in compare_prompts:
        prompts.append(
            ImaginePrompt(
                prompt_text,
                seed=1,
                model_weights=model_version,
            )
        )

    threshold = 35000
    results = list(imagine(prompts))
    for i, result in enumerate(results):
        img_path = f"{filename_base_for_orig_outputs}_{result.prompt.prompt_text}_{result.prompt.model_weights.aliases[0]}.png"
        result.img.save(img_path)

    for i, result in enumerate(results):
        img_path = f"{filename_base_for_orig_outputs}_{result.prompt.prompt_text}_{result.prompt.model_weights.aliases[0]}.png"
        assert_image_similar_to_expectation(
            result.img, img_path=img_path, threshold=threshold
        )


def test_img2img_beach_to_sunset(
    solver_type, filename_base_for_outputs, filename_base_for_orig_outputs
):
    img = LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/beach_at_sainte_adresse.jpg")
    prompt = ImaginePrompt(
        "a painting of beautiful cloudy sunset at the beach",
        init_image=img,
        init_image_strength=0.5,
        prompt_strength=15,
        mask_prompt="(sky|clouds) AND !(buildings|trees)",
        mask_mode="replace",
        size=512,
        steps=40 * 2,
        seed=1,
        solver_type=solver_type,
    )
    result = next(imagine(prompt))

    pillow_fit_image_within(img).save(f"{filename_base_for_orig_outputs}__orig.jpg")
    img_path = f"{filename_base_for_outputs}.png"
    assert_image_similar_to_expectation(result.img, img_path=img_path, threshold=2900)


def test_img_to_img_from_url_cats(
    solver_type,
    filename_base_for_outputs,
    mocked_responses,
    filename_base_for_orig_outputs,
    default_model_loaded,
):
    with open(
        os.path.join(TESTS_FOLDER, "data", "val2017-000000039769-cococats.jpg"), "rb"
    ) as f:
        img_data = f.read()
    mocked_responses.get(
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        body=img_data,
        status=200,
        content_type="image/jpeg",
    )
    img = LazyLoadingImage(url="http://images.cocodataset.org/val2017/000000039769.jpg")

    prompt = ImaginePrompt(
        "dogs lying on a hot pink couch",
        init_image=img,
        init_image_strength=0.5,
        size=512,
        steps=50,
        seed=1,
        solver_type=solver_type,
    )

    result = next(imagine(prompt))

    img = pillow_fit_image_within(img)
    img.save(f"{filename_base_for_orig_outputs}__orig.jpg")
    img_path = f"{filename_base_for_outputs}.png"
    assert_image_similar_to_expectation(result.img, img_path=img_path, threshold=17000)


def test_img2img_low_noise(
    filename_base_for_outputs,
    solver_type,
):
    fruit_path = os.path.join(TESTS_FOLDER, "data", "bowl_of_fruit.jpg")
    img = LazyLoadingImage(filepath=fruit_path)

    prompt = ImaginePrompt(
        "a white bowl filled with gold coins",
        prompt_strength=12,
        init_image=img,
        init_image_strength=0.5,
        mask_prompt="(fruit{*2} OR stem{*10} OR fruit stem{*3})",
        mask_mode="replace",
        # steps=40,
        seed=1,
        solver_type=solver_type,
    )

    result = next(imagine(prompt))

    threshold_lookup = {
        "dpmpp": 26000,
        "k_dpm_2_a": 26000,
        "k_euler_a": 18000,
        "k_dpm_adaptive": 13000,
    }
    threshold = threshold_lookup.get(solver_type, 14000)

    img_path = f"{filename_base_for_outputs}.png"
    assert_image_similar_to_expectation(
        result.img, img_path=img_path, threshold=threshold
    )


@pytest.mark.parametrize("init_strength", [0, 0.05, 0.2, 1])
def test_img_to_img_fruit_2_gold(
    filename_base_for_outputs,
    solver_type,
    init_strength,
    filename_base_for_orig_outputs,
):
    img = LazyLoadingImage(
        filepath=os.path.join(TESTS_FOLDER, "data", "bowl_of_fruit.jpg")
    )
    target_steps = 25
    needed_steps = 25 if init_strength >= 1 else int(target_steps / (1 - init_strength))
    prompt = ImaginePrompt(
        "a white bowl filled with gold coins",
        prompt_strength=12,
        init_image=img,
        init_image_strength=init_strength,
        mask_prompt="(fruit{*2} OR stem{*10} OR fruit stem{*3})",
        mask_mode="replace",
        steps=needed_steps,
        seed=1,
        solver_type=solver_type,
    )

    result = next(imagine(prompt))

    threshold_lookup = {
        "k_dpm_2_a": 32000,
        "k_euler_a": 18000,
        "k_dpm_adaptive": 13000,
        "k_dpmpp_2s": 16000,
    }
    threshold = threshold_lookup.get(solver_type, 16000)

    pillow_fit_image_within(img).save(f"{filename_base_for_orig_outputs}__orig.jpg")
    img_path = f"{filename_base_for_outputs}.png"
    assert_image_similar_to_expectation(
        result.img, img_path=img_path, threshold=threshold
    )


@pytest.mark.skipif(get_device() == "cpu", reason="Too slow to run on CPU")
def test_img_to_img_fruit_2_gold_repeat():
    img = LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/bowl_of_fruit.jpg")
    run_count = 1

    kwargs = {
        "prompt": "a white bowl filled with gold coins. sharp focus",
        "prompt_strength": 12,
        "init_image": img,
        "init_image_strength": 0.2,
        "mask_prompt": "(fruit OR stem{*5} OR fruit stem)",
        "mask_mode": "replace",
        "steps": 20,
        "seed": 946188797,
        "fix_faces": True,
        "upscale": True,
    }
    prompts = [
        ImaginePrompt(**kwargs),
        ImaginePrompt(**kwargs),
        ImaginePrompt(**kwargs),
    ]
    for result in imagine(prompts, debug_img_callback=None):
        result.img.save(
            f"{TESTS_FOLDER}/test_output/img2img_fruit_2_gold_{result.prompt.solver_type}_{get_device()}_run-{run_count:02}.jpg"
        )
        run_count += 1


@pytest.mark.skipif(get_device() == "cpu", reason="Too slow to run on CPU")
def test_img_to_file():
    prompt = ImaginePrompt(
        "an old growth forest, diffuse light poking through the canopy. high-resolution, nature photography, nat geo photo",
        size=(512 + 64, 512 - 64),
        steps=2,
        seed=2,
        upscale=True,
    )
    out_folder = f"{TESTS_FOLDER}/test_output"
    imagine_image_files(prompt, outdir=out_folder)


@pytest.mark.skipif(get_device() == "cpu", reason="Too slow to run on CPU")
def test_inpainting_bench(filename_base_for_outputs, filename_base_for_orig_outputs):
    img = LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/bench2.png")
    prompt = ImaginePrompt(
        "a wise old man",
        init_image=img,
        init_image_strength=0.4,
        mask_image=LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/bench2_mask.png"),
        size=512,
        steps=40,
        seed=1,
    )
    result = next(imagine(prompt))

    pillow_fit_image_within(img).save(f"{filename_base_for_orig_outputs}_orig.jpg")
    img_path = f"{filename_base_for_outputs}.png"
    assert_image_similar_to_expectation(result.img, img_path=img_path, threshold=2800)


@pytest.mark.skipif(get_device() == "cpu", reason="Too slow to run on CPU")
def test_cliptext_inpainting_pearl_doctor(
    filename_base_for_outputs, filename_base_for_orig_outputs
):
    img = LazyLoadingImage(
        filepath=f"{TESTS_FOLDER}/data/girl_with_a_pearl_earring.jpg"
    )
    prompt = ImaginePrompt(
        "a female doctor in the hospital",
        prompt_strength=12,
        init_image=img,
        init_image_strength=0.2,
        mask_prompt="face AND NOT (bandana OR hair OR blue fabric){*5}",
        mask_mode=MaskMode.KEEP,
        size=512,
        steps=40,
        seed=181509347,
    )
    result = next(imagine(prompt))

    pillow_fit_image_within(img).save(f"{filename_base_for_orig_outputs}_orig.jpg")
    img_path = f"{filename_base_for_outputs}.png"
    assert_image_similar_to_expectation(result.img, img_path=img_path, threshold=32000)


@pytest.mark.skipif(get_device() == "cpu", reason="Too slow to run on CPU")
def test_tile_mode(filename_base_for_outputs):
    prompt_text = "gold coins"
    prompt = ImaginePrompt(
        prompt_text,
        size=400,
        steps=15,
        seed=1,
        tile_mode="xy",
    )
    result = next(imagine(prompt))

    img_path = f"{filename_base_for_outputs}.png"
    assert_image_similar_to_expectation(result.img, img_path=img_path, threshold=26000)


control_modes = list(CONTROL_MODES.keys())


@pytest.mark.parametrize("control_mode", control_modes)
@pytest.mark.gputest
def test_controlnet(filename_base_for_outputs, control_mode):
    prompt_text = "a photo of a woman sitting on a bench"
    img = LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/bench2.png")
    control_input = ControlInput(
        mode=control_mode,
        image=img,
    )

    seed = 0
    if control_mode == "inpaint":
        prompt_text = "a wise old man"
        seed = 1
        mask_image = LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/bench2_mask.png")
        control_input = ControlInput(
            mode=control_mode,
            image=mask_image,
        )
    elif control_mode == "qrcode":
        prompt_text = "a fruit salad"
        swirl_img = LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/swirl.jpeg")
        control_input = ControlInput(
            mode=control_mode,
            image=swirl_img,
        )

    prompt = ImaginePrompt(
        prompt_text,
        size=512,
        steps=45,
        seed=seed,
        init_image=img,
        init_image_strength=0,
        control_inputs=[control_input],
        fix_faces=True,
        solver_type="ddim",
    )
    prompt.steps = 1
    prompt.size = 256
    result = next(imagine(prompt))
    prompt.steps = 15
    prompt.size = 512
    result = next(imagine(prompt))

    img_path = f"{filename_base_for_outputs}.png"
    assert_image_similar_to_expectation(result.img, img_path=img_path, threshold=25000)


@pytest.mark.skipif(
    get_device() in {"cpu", "mps"},
    reason="Too slow to run on CPU. Too much memory for MPS",
)
def test_large_image(filename_base_for_outputs):
    prompt_text = "a stormy ocean. oil painting"
    prompt = ImaginePrompt(
        prompt_text,
        size="1080p",
        steps=30,
        seed=0,
    )
    result = next(imagine(prompt))

    img_path = f"{filename_base_for_outputs}.png"
    assert_image_similar_to_expectation(result.img, img_path=img_path, threshold=35000)


@pytest.mark.parametrize(
    ("size", "tile_size", "stride", "expected"),
    [
        (128, 128, 64, [0]),  # exactly one tile
        (
            64,
            128,
            64,
            [0],
        ),  # smaller than tile — single pass, tile gets clamped by slicing
        (192, 128, 64, [0, 64]),  # two tiles with overlap
        (256, 128, 64, [0, 64, 128]),  # three tiles
        (135, 128, 64, [0, 7]),  # 1080p height in latent (1080/8)
        (240, 128, 64, [0, 64, 112]),  # 1920px width in latent
    ],
)
def test_get_tile_positions(size, tile_size, stride, expected):
    from imaginairy.api.generate_refiners import _get_tile_positions

    result = _get_tile_positions(size, tile_size, stride)
    assert result == expected
    assert result[0] == 0
    # when size >= tile_size, every tile must fit and cover the full extent
    if size >= tile_size:
        for pos in result:
            assert pos + tile_size <= size
        assert result[-1] + tile_size == size


def test_caption_tile_regions(monkeypatch):
    """Per-tile captioning: correct tile count, dedup of identical captions."""
    from unittest.mock import MagicMock

    from PIL import Image

    from imaginairy.api.generate_refiners import (
        MULTIDIFFUSION_STRIDE,
        MULTIDIFFUSION_TILE_SIZE,
        _caption_tile_regions,
        _get_tile_positions,
    )
    from imaginairy.schema import WeightedPrompt

    # Fake comp image large enough for tiling (1920x1080)
    comp_image = Image.new("RGB", (1920, 1080), color="blue")
    latent_h, latent_w = 1080 // 8, 1920 // 8  # 135, 240

    tile_ys = _get_tile_positions(
        latent_h, MULTIDIFFUSION_TILE_SIZE, MULTIDIFFUSION_STRIDE
    )
    tile_xs = _get_tile_positions(
        latent_w, MULTIDIFFUSION_TILE_SIZE, MULTIDIFFUSION_STRIDE
    )
    expected_tile_count = len(tile_ys) * len(tile_xs)

    # Monkeypatch BLIP to return a fixed caption per crop
    captions_returned = []

    def fake_caption(img, min_length=30):
        # Return same caption for all tiles to test dedup
        captions_returned.append(True)
        return "a blue background"

    monkeypatch.setattr(
        "imaginairy.enhancers.describe_image_blip.generate_caption", fake_caption
    )

    # Mock sd with a fake calculate_text_conditioning_kwargs
    import torch

    fake_conditioning = {"clip_text_embedding": torch.zeros(1, 77, 768)}
    sd = MagicMock()
    sd.calculate_text_conditioning_kwargs.return_value = fake_conditioning
    sd.unet.device = torch.device("cpu")
    sd.unet.dtype = torch.float32

    result = _caption_tile_regions(
        comp_image=comp_image,
        tile_ys=tile_ys,
        tile_xs=tile_xs,
        tile_size=MULTIDIFFUSION_TILE_SIZE,
        downsampling_factor=8,
        global_prompts=[WeightedPrompt(text="a mountain landscape")],
        negative_prompts=[WeightedPrompt(text="")],
        sd=sd,
    )

    # Should have conditioning for every tile
    assert len(result) == expected_tile_count
    assert len(captions_returned) == expected_tile_count

    # All captions identical → calculate_text_conditioning_kwargs called once (dedup)
    assert sd.calculate_text_conditioning_kwargs.call_count == 1

    # All tiles share the same dict object
    values = list(result.values())
    for v in values[1:]:
        assert v is values[0]
