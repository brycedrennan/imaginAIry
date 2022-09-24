import pytest

from imaginairy import LazyLoadingImage
from imaginairy.api import imagine, imagine_image_files
from imaginairy.schema import ImaginePrompt
from imaginairy.utils import get_device

from . import TESTS_FOLDER

device_sampler_type_test_cases = {
    "mps:0": [
        ("plms", "b4b434ed45919f3505ac2be162791c71"),
        ("ddim", "b369032a025915c0a7ccced165a609b3"),
        ("k_lms", "b87325c189799d646ccd07b331564eb6"),
        ("k_dpm_2", "cb37ca934938466bdbc1dd995da037de"),
        ("k_dpm_2_a", "ef155995ca1638f0ae7db9f573b83767"),
        ("k_euler", "d126da5ca8b08099cde8b5037464e788"),
        ("k_euler_a", "cac5ca2e26c31a544b76a9442eb2ea37"),
        ("k_heun", "0382ef71d9967fefd15676410289ebab"),
    ],
    "cuda": [
        ("plms", "0c44d2c8222f519a6700ebae54450435"),
        ("ddim", "4493ca85c2b24879525eac2b73e5a538"),
        ("k_lms", "82b38a5638a572d5968422b02e625f66"),
        ("k_dpm_2", "9df2fcd6256ff68c6cc4a6c603ae8f2e"),
        ("k_dpm_2_a", "0c5491c1a73094540ed15785f4106bca"),
        ("k_euler", "c82f628217fab06d8b5d5227827c1d92"),
        ("k_euler_a", "74f748a8371c2fcec54ecc5dcf1dbb64"),
        ("k_heun", "9ae586a7a8b10a0a0bf120405e4937e9"),
    ],
    "cpu": [],
}
sampler_type_test_cases = device_sampler_type_test_cases[get_device()]


@pytest.mark.parametrize("sampler_type,expected_md5", sampler_type_test_cases)
def test_imagine(sampler_type, expected_md5):
    prompt_text = "a scenic landscape"
    prompt = ImaginePrompt(
        prompt_text, width=512, height=256, steps=5, seed=1, sampler_type=sampler_type
    )
    result = next(imagine(prompt))
    result.img.save(
        f"{TESTS_FOLDER}/test_output/sampler_type_{sampler_type.upper()}.jpg"
    )
    assert result.md5() == expected_md5


device_sampler_type_test_cases_img_2_img = {
    "mps:0": {
        ("plms", "0d9c40c348cdac7bdc8d5a472f378f42"),
        ("ddim", "12921ee5a8d276f1b477d196d304fef2"),
    },
    "cuda": {
        ("plms", "28752d4e1d778abc3e9424f4f23d1aaf"),
        ("ddim", "28752d4e1d778abc3e9424f4f23d1aaf"),
    },
    "cpu": [],
}
sampler_type_test_cases_img_2_img = device_sampler_type_test_cases_img_2_img[
    get_device()
]


@pytest.mark.skipif(get_device() == "cpu", reason="Too slow to run on CPU")
@pytest.mark.parametrize("sampler_type,expected_md5", sampler_type_test_cases_img_2_img)
def test_img_to_img(sampler_type, expected_md5):
    prompt = ImaginePrompt(
        "a photo of a beach",
        init_image=f"{TESTS_FOLDER}/data/beach_at_sainte_adresse.jpg",
        init_image_strength=0.8,
        width=512,
        height=512,
        steps=5,
        seed=1,
        sampler_type=sampler_type,
    )
    result = next(imagine(prompt))
    result.img.save(
        f"{TESTS_FOLDER}/test_output/sampler_type_{sampler_type.upper()}_img2img_beach.jpg"
    )
    assert result.md5() == expected_md5


@pytest.mark.skipif(get_device() == "cpu", reason="Too slow to run on CPU")
def test_img_to_img_from_url():
    prompt = ImaginePrompt(
        "dogs lying on a hot pink couch",
        init_image=LazyLoadingImage(
            url="http://images.cocodataset.org/val2017/000000039769.jpg"
        ),
        init_image_strength=0.5,
        width=512,
        height=512,
        steps=5,
        seed=1,
        sampler_type="DDIM",
    )
    out_folder = f"{TESTS_FOLDER}/test_output"
    imagine_image_files(prompt, outdir=out_folder)


@pytest.mark.skipif(get_device() == "cpu", reason="Too slow to run on CPU")
def test_img_to_file():
    prompt = ImaginePrompt(
        "an old growth forest, diffuse light poking through the canopy. high-resolution, nature photography, nat geo photo",
        width=512 + 64,
        height=512 - 64,
        steps=5,
        seed=2,
        sampler_type="PLMS",
        upscale=True,
    )
    out_folder = f"{TESTS_FOLDER}/test_output"
    imagine_image_files(prompt, outdir=out_folder)


@pytest.mark.skipif(get_device() == "cpu", reason="Too slow to run on CPU")
def test_inpainting():
    prompt = ImaginePrompt(
        "a basketball on a bench",
        init_image=f"{TESTS_FOLDER}/data/bench2.png",
        init_image_strength=0.4,
        mask_image=LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/bench2_mask.png"),
        width=512,
        height=512,
        steps=5,
        seed=1,
        sampler_type="DDIM",
    )
    out_folder = f"{TESTS_FOLDER}/test_output"
    imagine_image_files(prompt, outdir=out_folder)


@pytest.mark.skipif(get_device() == "cpu", reason="Too slow to run on CPU")
def test_cliptext_inpainting():
    prompts = [
        ImaginePrompt(
            "elegant woman. oil painting",
            prompt_strength=12,
            init_image=f"{TESTS_FOLDER}/data/girl_with_a_pearl_earring.jpg",
            init_image_strength=0.3,
            mask_prompt="face{*2}",
            mask_mode=ImaginePrompt.MaskMode.KEEP,
            width=512,
            height=512,
            steps=5,
            sampler_type="DDIM",
        ),
    ]
    out_folder = f"{TESTS_FOLDER}/test_output"
    imagine_image_files(prompts, outdir=out_folder)
