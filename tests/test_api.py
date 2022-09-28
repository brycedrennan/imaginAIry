import os.path

import pytest

from imaginairy import LazyLoadingImage
from imaginairy.api import imagine, imagine_image_files, prompt_normalized
from imaginairy.img_utils import pillow_fit_image_within
from imaginairy.schema import ImaginePrompt
from imaginairy.utils import get_device

from . import TESTS_FOLDER

device_sampler_type_test_cases = {
    "mps:0": [
        ("plms", "78539ae3a3097dc8232da6d630551ab3"),
        ("ddim", "828fc143cd40586347b2f8403c288c9b"),
        ("k_lms", "53d25e59add39c8447537be30e4eff4b"),
        ("k_dpm_2", "5108bceb58a38d88a585f37b2ba1b072"),
        ("k_dpm_2_a", "20396daa6c920d1cfd6db90e73558c01"),
        ("k_euler", "9ab4666ebe6c3aa68673912bb17fb2b1"),
        ("k_euler_a", "c4b03829cc93422801f3243a46bad4bc"),
        ("k_heun", "0d3aad6800d4a9a43f0b0514af9d23b5"),
    ],
    "cuda": [
        ("plms", "b98e1248ad1f144d34122d8809b39fb8"),
        ("ddim", "a645ca24575ed3f18bf48f11354233bb"),
        ("k_lms", "3ddbdef45e3f38768730961771d01727"),
        ("k_dpm_2", "b6e88e16ec2c43e6382b1adec828479d"),
        ("k_dpm_2_a", "b0791770d48cb22d308ad76c72fb660f"),
        ("k_euler", "bcf375769d64d9ca224864d35565ac1d"),
        ("k_euler_a", "38b970ff6a67428efbf00df66a9e48f7"),
        ("k_heun", "ccbd0804c7ce2bb637c682951bd8b693"),
    ],
    "cpu": [],
}
sampler_type_test_cases = device_sampler_type_test_cases[get_device()]


@pytest.mark.parametrize("sampler_type,expected_md5", sampler_type_test_cases)
def test_imagine(sampler_type, expected_md5, filename_base_for_outputs):
    prompt_text = "a scenic landscape"
    prompt = ImaginePrompt(
        prompt_text, width=512, height=256, steps=20, seed=1, sampler_type=sampler_type
    )
    result = next(imagine(prompt))
    result.img.save(f"{filename_base_for_outputs}.jpg")
    assert result.md5() == expected_md5


device_sampler_type_test_cases_img_2_img = {
    "mps:0": {
        ("plms", "0d9c40c348cdac7bdc8d5a472f378f42"),
        ("ddim", "0d9c40c348cdac7bdc8d5a472f378f42"),
    },
    "cuda": {
        ("plms", "841723966344dd8678aee1ce5f9cbb3d"),
        ("ddim", "1f0d72370fabcf2ff716e4068d5b2360"),
    },
}
sampler_type_test_cases_img_2_img = device_sampler_type_test_cases_img_2_img.get(
    get_device(), []
)


@pytest.mark.skipif(get_device() == "cpu", reason="Too slow to run on CPU")
@pytest.mark.parametrize("sampler_type,expected_md5", sampler_type_test_cases_img_2_img)
def test_img2img_beach_to_sunset(sampler_type, expected_md5, filename_base_for_outputs):
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
    img = pillow_fit_image_within(img)
    img.save(f"{filename_base_for_outputs}__orig.jpg")
    result.img.save(f"{filename_base_for_outputs}.jpg")


device_sampler_type_test_cases_img_2_img = {
    "mps:0": {
        ("plms", "e9bb714771f7984e61debabc4bb3cd22"),
        ("ddim", "62bacc4ae391e6775a3723c88738ec61"),
    },
    "cuda": {
        ("plms", "b8c7b52da977c1531a9a61c0a082404c"),
        ("ddim", "d6784710dd78e4cb628aba28322b04cf"),
    },
}
sampler_type_test_cases_img_2_img = device_sampler_type_test_cases_img_2_img.get(
    get_device(), []
)


@pytest.mark.skipif(get_device() == "cpu", reason="Too slow to run on CPU")
@pytest.mark.parametrize("sampler_type,expected_md5", sampler_type_test_cases_img_2_img)
def test_img_to_img_from_url_cats(
    sampler_type, expected_md5, filename_base_for_outputs
):
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
    img.save(f"{filename_base_for_outputs}__orig.jpg")
    result.img.save(f"{filename_base_for_outputs}.jpg")

    assert result.md5() == expected_md5


@pytest.mark.skipif(get_device() == "cpu", reason="Too slow to run on CPU")
@pytest.mark.parametrize("sampler_type", ["ddim", "plms"])
@pytest.mark.parametrize("init_strength", [0, 0.05, 0.2, 1])
def test_img_to_img_fruit_2_gold(
    filename_base_for_outputs, sampler_type, init_strength
):
    img = LazyLoadingImage(
        url="https://raw.githubusercontent.com/brycedrennan/imaginAIry/master/assets/000056_293284644_PLMS40_PS7.5_photo_of_a_bowl_of_fruit.jpg"
    )

    prompt = ImaginePrompt(
        "a white bowl filled with gold coins",
        prompt_strength=12,
        init_image=img,
        init_image_strength=init_strength,
        mask_prompt="(fruit OR stem{*5} OR fruit stem)",
        mask_mode="replace",
        steps=80,
        seed=1,
        sampler_type=sampler_type,
    )

    result = next(imagine(prompt))

    img = pillow_fit_image_within(img)
    img.save(f"{filename_base_for_outputs}__orig.jpg")
    result.img.save(f"{filename_base_for_outputs}.jpg")


@pytest.mark.skipif(get_device() == "cpu", reason="Too slow to run on CPU")
def test_img_to_img_fruit_2_gold_repeat():
    """Run this test manually to"""
    img = LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/bowl_of_fruit.jpg")
    outdir = f"{TESTS_FOLDER}/test_output/"
    run_count = 1

    def _record_step(img, description, step_count, prompt):
        steps_path = os.path.join(
            outdir,
            f"steps_fruit_2_gold_repeat_{get_device()}_S{prompt.seed}_run_{run_count:02}",
        )
        os.makedirs(steps_path, exist_ok=True)
        filename = f"fruit_2_gold_repeat_{get_device()}_S{prompt.seed}_step{step_count:04}_{prompt_normalized(description)[:40]}.jpg"

        destination = os.path.join(steps_path, filename)
        img.save(destination)

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
        img = pillow_fit_image_within(img)
        img.save(f"{TESTS_FOLDER}/test_output/img2img_fruit_2_gold__orig.jpg")
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
def test_inpainting_bench(filename_base_for_outputs):
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

    img = pillow_fit_image_within(img)
    img.save(f"{filename_base_for_outputs}__orig.jpg")
    result.img.save(f"{filename_base_for_outputs}.jpg")


@pytest.mark.skipif(get_device() == "cpu", reason="Too slow to run on CPU")
def test_cliptext_inpainting_pearl_doctor(filename_base_for_outputs):
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

    img = pillow_fit_image_within(img)
    img.save(f"{filename_base_for_outputs}__orig.jpg")
    result.img.save(f"{filename_base_for_outputs}_{prompt.seed}_01.jpg")
