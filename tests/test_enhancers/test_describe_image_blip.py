import pytest
from PIL import Image

from imaginairy.enhancers.describe_image_blip import generate_caption
from imaginairy.utils import get_device, seed_everything
from tests import TESTS_FOLDER


@pytest.mark.skipif(get_device() == "cpu", reason="Too slow to run on CPU")
def test_describe_picture():
    seed_everything(1)
    img = Image.open(f"{TESTS_FOLDER}/data/girl_with_a_pearl_earring.jpg")
    caption = generate_caption(img)
    assert caption in {
        "a painting of a girl with a pearl earring wearing a yellow dress and a pearl earring in her ear and a black background",
        "a painting of a girl with a pearl ear wearing a yellow dress and a pearl earring on her left ear and a black background",
        "a painting of a woman with a pearl ear wearing an ornament pearl earring and wearing an orange, white, blue and yellow dress",
        "a painting of a woman with a pearl earring looking to her left, in profile with her right eye partially closed, standing upright",
    }
