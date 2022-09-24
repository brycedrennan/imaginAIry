from PIL import Image

from imaginairy.safety import is_nsfw
from tests import TESTS_FOLDER


def test_is_nsfw():
    img = Image.open(f"{TESTS_FOLDER}/data/safety.jpg")
    assert is_nsfw(img)

    img = Image.open(f"{TESTS_FOLDER}/data/girl_with_a_pearl_earring.jpg")
    assert not is_nsfw(img)
