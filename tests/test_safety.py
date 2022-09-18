from PIL import Image

from imaginairy.api import load_model
from imaginairy.safety import is_nsfw
from imaginairy.utils import get_device, pillow_img_to_torch_image
from tests import TESTS_FOLDER


def test_is_nsfw():
    img = Image.open(f"{TESTS_FOLDER}/data/safety.jpg")
    latent = _pil_to_latent(img)
    assert is_nsfw(img, latent)

    img = Image.open(f"{TESTS_FOLDER}/data/girl_with_a_pearl_earring.jpg")
    latent = _pil_to_latent(img)
    assert not is_nsfw(img, latent)


def _pil_to_latent(img):
    model = load_model()
    img = pillow_img_to_torch_image(img)
    img = img.to(get_device())
    latent = model.get_first_stage_encoding(model.encode_first_stage(img))
    return latent
