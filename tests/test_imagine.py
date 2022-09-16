import pytest

from imaginairy import LazyLoadingImage
from imaginairy.api import imagine, imagine_image_files
from imaginairy.schema import ImaginePrompt
from imaginairy.utils import get_device

from . import TESTS_FOLDER

device_sampler_type_test_cases = {
    "mps": {
        ("plms", "b4b434ed45919f3505ac2be162791c71"),
        ("ddim", "b369032a025915c0a7ccced165a609b3"),
        ("k_lms", "b87325c189799d646ccd07b331564eb6"),
        ("k_dpm_2", "cb37ca934938466bdbc1dd995da037de"),
        ("k_dpm_2_a", "ef155995ca1638f0ae7db9f573b83767"),
        ("k_euler", "d126da5ca8b08099cde8b5037464e788"),
        ("k_euler_a", "cac5ca2e26c31a544b76a9442eb2ea37"),
        ("k_heun", "0382ef71d9967fefd15676410289ebab"),
    },
    "cuda": {
        ("plms", "62e78287e7848e48d45a1b207fb84102"),
        ("ddim", "164c2a008b100e5fa07d3db2018605bd"),
        ("k_lms", "450fea507ccfb44b677d30fae9f40a52"),
        ("k_dpm_2", "901daad7a9e359404d8e3d3f4236c4ce"),
        ("k_dpm_2_a", "855e80286dfdc89752f6bdd3fdeb1a62"),
        ("k_euler", "06df9c19d472bfa6530db98be4ea10e8"),
        ("k_euler_a", "79552628ff77914c8b6870703fe116b5"),
        ("k_heun", "8ced3578ae25d34da9f4e4b1a20bf416"),
    },
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


def test_img_to_img():
    prompt = ImaginePrompt(
        "a photo of a beach",
        init_image=f"{TESTS_FOLDER}/data/beach_at_sainte_adresse.jpg",
        init_image_strength=0.8,
        width=512,
        height=512,
        steps=5,
        seed=1,
        sampler_type="DDIM",
    )
    out_folder = f"{TESTS_FOLDER}/test_output"
    imagine_image_files(prompt, outdir=out_folder)


def test_img_to_img_from_url():
    prompt = ImaginePrompt(
        "dogs lying on a hot pink couch",
        init_image=LazyLoadingImage(
            url="http://images.cocodataset.org/val2017/000000039769.jpg"
        ),
        init_image_strength=0.5,
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
        steps=5,
        seed=2,
        sampler_type="PLMS",
        upscale=True,
    )
    out_folder = f"{TESTS_FOLDER}/test_output"
    imagine_image_files(prompt, outdir=out_folder)
