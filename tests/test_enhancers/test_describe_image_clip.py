import pytest
from PIL import Image

from imaginairy.enhancers.describe_image_clip import find_img_text_similarity
from imaginairy.utils import get_device
from tests import TESTS_FOLDER


@pytest.mark.skipif(get_device() == "cpu", reason="Too slow to run on CPU")
def test_clip_text_comparison():
    img = Image.open(f"{TESTS_FOLDER}/data/girl_with_a_pearl_earring.jpg")
    phrases = [
        "Johannes Vermeer painting",
        "a painting of a girl with a pearl earring",
        "a bulldozer",
        "photo",
    ]
    probs = find_img_text_similarity(img, phrases)
    assert probs[:2] == [
        (
            "a painting of a girl with a pearl earring",
            pytest.approx(0.2857227921485901, abs=0.01),
        ),
        ("Johannes Vermeer painting", pytest.approx(0.25186583399772644, abs=0.01)),
    ]
