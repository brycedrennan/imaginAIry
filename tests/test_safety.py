import pytest
from PIL import Image

from imaginairy.utils import get_device
from imaginairy.utils.safety import create_safety_score
from tests import TESTS_FOLDER


@pytest.mark.skipif(get_device() == "cpu", reason="Too slow to run on CPU")
def test_is_nsfw():
    img = Image.open(f"{TESTS_FOLDER}/data/safety.jpg")

    safety_score = create_safety_score(img)
    assert safety_score.is_nsfw

    img = Image.open(f"{TESTS_FOLDER}/data/girl_with_a_pearl_earring.jpg")
    safety_score = create_safety_score(img)
    assert not safety_score.is_nsfw

    img = Image.open(f"{TESTS_FOLDER}/data/black_square.jpg")

    safety_score = create_safety_score(img)
    assert not safety_score.is_nsfw
