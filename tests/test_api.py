import os.path

import pytest

from imaginairy import LazyLoadingImage
from imaginairy.api import imagine, imagine_image_files
from imaginairy.img_utils import pillow_fit_image_within
from imaginairy.schema import ImaginePrompt
from imaginairy.utils import get_device

from . import TESTS_FOLDER
from .utils import assert_image_similar_to_expectation


def test_imagine(sampler_type, filename_base_for_outputs):
    prompt_text = "a scenic old-growth forest with diffuse light poking through the canopy. high resolution nature photography"
    prompt = ImaginePrompt(
        prompt_text, width=512, height=512, steps=20, seed=1, sampler_type=sampler_type
    )
    result = next(imagine(prompt))

    threshold_lookup = {
        "k_dpm_2_a": 26000
    }
    threshold = threshold_lookup.get(sampler_type, 10000)

    img_path = f"{filename_base_for_outputs}.png"
    assert_image_similar_to_expectation(result.img, img_path=img_path, threshold=threshold)


def test_img2img_beach_to_sunset(
    sampler_type, filename_base_for_outputs, filename_base_for_orig_outputs
):
    img = LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/beach_at_sainte_adresse.jpg")
    prompt = ImaginePrompt(
        "a painting of beautiful cloudy sunset at the beach",
        init_image=img,
        init_image_strength=0.5,
        prompt_strength=15,
        mask_prompt="(sky|clouds) AND !(buildings|trees)",
        mask_mode="replace",
        width=512,
        height=512,
        steps=40 * 2,
        seed=1,
        sampler_type=sampler_type,
    )
    result = next(imagine(prompt))

    pillow_fit_image_within(img).save(f"{filename_base_for_orig_outputs}__orig.jpg")
    img_path = f"{filename_base_for_outputs}.png"
    assert_image_similar_to_expectation(result.img, img_path=img_path, threshold=2800)


def test_img_to_img_from_url_cats(
    sampler_type,
    filename_base_for_outputs,
    mocked_responses,
    filename_base_for_orig_outputs,
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
        width=512,
        height=512,
        steps=50,
        seed=1,
        sampler_type=sampler_type,
    )

    result = next(imagine(prompt))

    img = pillow_fit_image_within(img)
    img.save(f"{filename_base_for_orig_outputs}__orig.jpg")
    img_path = f"{filename_base_for_outputs}.png"
    assert_image_similar_to_expectation(result.img, img_path=img_path, threshold=14000)


@pytest.mark.parametrize("init_strength", [0, 0.05, 0.2, 1])
def test_img_to_img_fruit_2_gold(
    filename_base_for_outputs,
    sampler_type,
    init_strength,
    filename_base_for_orig_outputs,
):
    img = LazyLoadingImage(
        filepath=os.path.join(TESTS_FOLDER, "data", "bowl_of_fruit.jpg")
    )

    prompt = ImaginePrompt(
        "a white bowl filled with gold coins",
        prompt_strength=12,
        init_image=img,
        init_image_strength=init_strength,
        mask_prompt="(fruit{*2} OR stem{*10} OR fruit stem{*3})",
        mask_mode="replace",
        steps=80,
        seed=1,
        sampler_type=sampler_type,
    )

    result = next(imagine(prompt))

    threshold_lookup = {
        "k_dpm_2_a": 26000
    }
    threshold = threshold_lookup.get(sampler_type, 10000)

    pillow_fit_image_within(img).save(f"{filename_base_for_orig_outputs}__orig.jpg")
    img_path = f"{filename_base_for_outputs}.png"
    assert_image_similar_to_expectation(result.img, img_path=img_path, threshold=threshold)


@pytest.mark.skipif(get_device() == "cpu", reason="Too slow to run on CPU")
def test_img_to_img_fruit_2_gold_repeat():
    img = LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/bowl_of_fruit.jpg")
    run_count = 1

    kwargs = dict(
        prompt="a white bowl filled with gold coins. sharp focus",
        prompt_strength=12,
        init_image=img,
        init_image_strength=0.2,
        mask_prompt="(fruit OR stem{*5} OR fruit stem)",
        mask_mode="replace",
        steps=20,
        seed=946188797,
        sampler_type="plms",
        fix_faces=True,
        upscale=True,
    )
    prompts = [
        ImaginePrompt(**kwargs),
        ImaginePrompt(**kwargs),
        ImaginePrompt(**kwargs),
    ]
    for result in imagine(prompts, img_callback=None):
        result.img.save(
            f"{TESTS_FOLDER}/test_output/img2img_fruit_2_gold_plms_{get_device()}_run-{run_count:02}.jpg"
        )
        run_count += 1


@pytest.mark.skipif(get_device() == "cpu", reason="Too slow to run on CPU")
def test_img_to_file():
    prompt = ImaginePrompt(
        "an old growth forest, diffuse light poking through the canopy. high-resolution, nature photography, nat geo photo",
        width=512 + 64,
        height=512 - 64,
        steps=20,
        seed=2,
        sampler_type="PLMS",
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
        width=512,
        height=512,
        steps=40,
        seed=1,
        sampler_type="plms",
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
        mask_mode=ImaginePrompt.MaskMode.KEEP,
        width=512,
        height=512,
        steps=40,
        sampler_type="plms",
        seed=181509347,
    )
    result = next(imagine(prompt))

    pillow_fit_image_within(img).save(f"{filename_base_for_orig_outputs}_orig.jpg")
    img_path = f"{filename_base_for_outputs}.png"
    assert_image_similar_to_expectation(result.img, img_path=img_path, threshold=2800)
