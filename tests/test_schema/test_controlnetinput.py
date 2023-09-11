import pytest
from pydantic import ValidationError

from imaginairy import LazyLoadingImage
from imaginairy.schema import ControlNetInput
from tests import TESTS_FOLDER


@pytest.fixture(name="lazy_img")
def _lazy_img():
    return LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/red.png")


def test_controlnetinput_basic(lazy_img):
    ControlNetInput(mode="canny", image=lazy_img)
    ControlNetInput(mode="canny", image_raw=lazy_img)


def test_controlnetinput_invalid_mode(lazy_img):
    with pytest.raises(ValueError, match=r".*Invalid controlnet mode.*"):
        ControlNetInput(mode="pizza", image=lazy_img)


def test_controlnetinput_both_images(lazy_img):
    with pytest.raises(ValueError, match=r".*cannot specify both.*"):
        ControlNetInput(mode="canny", image=lazy_img, image_raw=lazy_img)


def test_controlnetinput_filepath_input(lazy_img):
    """Test that we accept filepaths here."""
    c = ControlNetInput(mode="canny", image=f"{TESTS_FOLDER}/data/red.png")
    c.image.convert("RGB")
    c = ControlNetInput(mode="canny", image_raw=f"{TESTS_FOLDER}/data/red.png")
    c.image_raw.convert("RGB")


def test_controlnetinput_big(lazy_img):
    with pytest.raises(ValidationError, match=r".*less than or.*"):
        ControlNetInput(mode="canny", strength=2**2048)
