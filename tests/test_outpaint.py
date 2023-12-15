import pytest

from imaginairy.api import imagine
from imaginairy.schema import ImaginePrompt, LazyLoadingImage
from imaginairy.utils import get_device
from imaginairy.utils.outpaint import outpaint_arg_str_parse
from tests import TESTS_FOLDER
from tests.utils import assert_image_similar_to_expectation


@pytest.mark.skipif(get_device() == "cpu", reason="Too slow to run on CPU")
def test_outpainting_outpaint(filename_base_for_outputs):
    img = LazyLoadingImage(
        filepath=f"{TESTS_FOLDER}/data/girl_with_a_pearl_earring.jpg"
    )
    prompt = ImaginePrompt(
        prompt="woman standing",
        init_image=img,
        init_image_strength=0,
        mask_prompt="background",
        outpaint="all250,up0,down600",
        mask_mode="replace",
        negative_prompt="picture frame, borders, framing, text, writing, watermarks, indoors, advertisement, paper, canvas, stock photo",
        steps=20,
        seed=542906833,
    )
    result = next(iter(imagine([prompt])))
    img_path = f"{filename_base_for_outputs}.png"
    assert_image_similar_to_expectation(result.img, img_path=img_path, threshold=17000)


outpaint_test_params = [
    ("A132", {"_all": 132}),
    ("A132,U50", {"_all": 132, "up": 50}),
    ("A132,U50,D50", {"_all": 132, "up": 50, "down": 50}),
    ("a132,u50,d50", {"_all": 132, "up": 50, "down": 50}),
    ("all50,up20,down600", {"_all": 50, "up": 20, "down": 600}),
]


@pytest.mark.parametrize(("arg_str", "expected_kwargs"), outpaint_test_params)
def test_outpaint_parse_kwargs(arg_str, expected_kwargs):
    assert outpaint_arg_str_parse(arg_str) == expected_kwargs
