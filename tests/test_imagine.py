import pytest

from imaginairy.api import imagine, imagine_image_files
from imaginairy.schema import ImaginePrompt
from imaginairy.utils import get_device

from . import TESTS_FOLDER

mps_sampler_type_test_cases = {
    ("plms", "3f211329796277a1870378288769fcde"),
    ("ddim", "70dbf2acce2c052e4e7f37412ae0366e"),
    ("k_lms", "3585c10c8f27bf091c15e761dca4d578"),
    ("k_dpm_2", "29b07125c9879540f8efac317ae33aea"),
    ("k_dpm_2_a", "4fd6767980444ca72e97cba2d0491eb4"),
    ("k_euler", "50609b279cff756db42ab9d2c85328ed"),
    ("k_euler_a", "ae7ac199c10f303e5ebd675109e59b23"),
    ("k_heun", "3668fe66770538337ac8c0b7ac210892"),
}


@pytest.mark.skipif(get_device() != "mps", reason="mps hashes")
@pytest.mark.parametrize("sampler_type,expected_md5", mps_sampler_type_test_cases)
def test_imagine(sampler_type, expected_md5):
    prompt_text = "a scenic landscape"
    prompt = ImaginePrompt(
        prompt_text, width=512, height=256, steps=10, seed=1, sampler_type=sampler_type
    )
    result = next(imagine(prompt))
    result.img.save(
        f"{TESTS_FOLDER}/test_output/sampler_type_{sampler_type.upper()}.jpg"
    )
    assert result.md5() == expected_md5


def test_img_to_img():
    prompt = ImaginePrompt(
        "a photo of a beach",
        init_image=f"{TESTS_FOLDER}/data/beach_at_sainte_adresse.jpg",
        init_image_strength=0.8,
        width=512,
        height=512,
        steps=50,
        seed=1,
        sampler_type="DDIM",
    )
    out_folder = f"{TESTS_FOLDER}/test_output"
    imagine_image_files(prompt, outdir=out_folder)


def test_img_to_file():
    prompt = ImaginePrompt(
        "an old growth forest, diffuse light poking through the canopy. high-resolution, nature photography, nat geo photo",
        width=512 + 64,
        height=512 - 64,
        steps=50,
        seed=2,
        sampler_type="PLMS",
        upscale=True,
    )
    out_folder = f"{TESTS_FOLDER}/test_output"
    imagine_image_files(prompt, outdir=out_folder)
