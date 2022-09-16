import pytest

from imaginairy import LazyLoadingImage
from imaginairy.schema import InvalidUrlError
from tests import TESTS_FOLDER


def test_lazy_load_image():
    with pytest.raises(ValueError, match=r".*specify a url or filepath.*"):
        LazyLoadingImage()

    with pytest.raises(FileNotFoundError, match=r".*File does not exist.*"):
        LazyLoadingImage(filepath="/tmp/bterpojirewpdfsn/ergqgr")

    with pytest.raises(InvalidUrlError):
        LazyLoadingImage(url="/tmp/bterpojirewpdfsn/ergqgr")

    img = LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/beach_at_sainte_adresse.jpg")
    assert img.size == (1686, 1246)
