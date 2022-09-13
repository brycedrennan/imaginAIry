from imaginairy.api import imagine, imagine_image_files
from imaginairy.schema import ImaginePrompt

from . import TESTS_FOLDER


def test_imagine():
    prompt = ImaginePrompt(
        "a scenic landscape", width=512, height=256, steps=20, seed=1
    )
    result = next(imagine(prompt))
    assert result.md5() == "4c5957c498881d365cfcf13014812af0"
    result.img.save(f"{TESTS_FOLDER}/test_output/scenic_landscape.png")


def test_img_to_img():
    prompt = ImaginePrompt(
        "a photo of a beach",
        init_image=f"{TESTS_FOLDER}/data/beach_at_sainte_adresse.jpg",
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
        steps=50,
        seed=2,
        sampler_type="PLMS",
        upscale=True,
    )
    out_folder = f"{TESTS_FOLDER}/test_output"
    imagine_image_files(prompt, outdir=out_folder)
