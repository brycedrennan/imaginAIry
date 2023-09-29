import pytest
from PIL import Image

from imaginairy.enhancers.blur_detect import is_blurry
from tests import TESTS_FOLDER

blur_params = [
    (f"{TESTS_FOLDER}/data/black_square.jpg", True),
    (f"{TESTS_FOLDER}/data/safety.jpg", False),
    (f"{TESTS_FOLDER}/data/latent_noise.jpg", False),
    (f"{TESTS_FOLDER}/data/girl_with_a_pearl_earring.jpg", False),
]


@pytest.mark.parametrize(("img_path", "expected"), blur_params)
def test_calculate_blurriness_level(img_path, expected):
    img = Image.open(img_path)

    assert is_blurry(img) == expected
