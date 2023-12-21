from PIL import Image

from imaginairy.enhancers.upscale_realesrgan import upscale_image
from tests import TESTS_FOLDER
from tests.utils import assert_image_similar_to_expectation


def test_upscale_textured_image(filename_base_for_outputs):
    img = Image.open(f"{TESTS_FOLDER}/data/sand_upscale_difficult.jpg")
    upscaled_image = upscale_image(img, ultrasharp=True)
    assert_image_similar_to_expectation(
        upscaled_image, f"{filename_base_for_outputs}.jpg", threshold=25000
    )
