import os.path
from typing import Optional

import pytest
from PIL import Image
from pydantic import BaseModel

from imaginairy.schema import InvalidUrlError, LazyLoadingImage
from tests import TESTS_FOLDER


class TestModel(BaseModel):
    header_img: Optional[LazyLoadingImage]


@pytest.fixture(name="red_url")
def _red_url(mocked_responses):
    url = "http://example.com/red.png"
    with open(os.path.join(TESTS_FOLDER, "data", "red.png"), "rb") as f:
        img_data = f.read()

    mocked_responses.get(
        url,
        body=img_data,
        status=200,
        content_type="image/png",
    )
    return url


@pytest.fixture(name="red_path")
def _red_path():
    return os.path.join(TESTS_FOLDER, "data", "red.png")


@pytest.fixture(name="red_b64")
def _red_b64():
    return "iVBORw0KGgoAAAANSUhEUgAAAgAAAAIAAQMAAADOtka5AAAABlBMVEX/AAD///9BHTQRAAAANklEQVR4nO3BAQEAAACCIP+vbkhAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB8G4IAAAHSeInwAAAAAElFTkSuQmCC"


def test_lazy_load_image(mocked_responses, red_url, red_path, red_b64):
    ll_img = LazyLoadingImage(filepath=red_path)
    assert ll_img.size == (512, 512)
    assert ll_img.as_base64() == red_b64

    ll_img = LazyLoadingImage(url=red_url)
    assert ll_img.size == (512, 512)
    assert ll_img.as_base64() == red_b64

    ll_img = LazyLoadingImage(img=Image.open(red_path))
    assert ll_img.size == (512, 512)
    assert ll_img.as_base64() == red_b64

    ll_img = LazyLoadingImage(b64=red_b64)
    assert ll_img.size == (512, 512)
    assert ll_img.as_base64() == red_b64


def test_lazy_load_image_validation():
    with pytest.raises(ValueError, match=r".*specify a url or filepath.*"):
        LazyLoadingImage()

    with pytest.raises(FileNotFoundError, match=r".*File does not exist.*"):
        LazyLoadingImage(filepath="/tmp/bterpojirewpdfsn/ergqgr")

    with pytest.raises(InvalidUrlError):
        LazyLoadingImage(url="/tmp/bterpojirewpdfsn/ergqgr")

    img = LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/beach_at_sainte_adresse.jpg")
    assert img.size == (1686, 1246)


def test_image_dump(red_path, red_b64):
    obj = TestModel(header_img=LazyLoadingImage(filepath=red_path))
    assert obj.header_img.size == (512, 512)

    obj_data = obj.model_dump_json()
    new_obj = TestModel.model_validate_json(obj_data)
    assert new_obj.header_img.size == (512, 512)
    assert new_obj.header_img.as_base64() == red_b64

    obj_data = obj.model_dump(mode="json")
    new_obj = TestModel.model_validate(obj_data)
    assert new_obj.header_img.size == (512, 512)
    assert new_obj.header_img.as_base64() == red_b64

    obj_data = obj.model_dump(mode="python")
    new_obj = TestModel.model_validate(obj_data)
    assert new_obj.header_img.size == (512, 512)
    assert new_obj.header_img.as_base64() == red_b64


def test_image_deserialization(red_path, red_url):
    rows = [
        {"header_img": LazyLoadingImage(filepath=red_path)},
        {"header_img": red_path},
        {"header_img": {"filepath": red_path}},
        {"header_img": {"url": red_url}},
    ]
    for row in rows:
        obj = TestModel.model_validate(row)
        assert obj.header_img.size == (512, 512)


def test_image_state(red_path):
    """I dont remember what this fixes. Maybe the ability of pydantic to copy an object?."""
    img = LazyLoadingImage(filepath=red_path)

    # bypass init
    img2 = LazyLoadingImage.__new__(LazyLoadingImage)
    img2.__setstate__(img.__getstate__())

    assert repr(img) == repr(img2)
