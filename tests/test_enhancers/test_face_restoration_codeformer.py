import pytest
from PIL import Image

from imaginairy.enhancers.face_restoration_codeformer import enhance_faces
from imaginairy.utils import get_device, seed_everything
from tests import TESTS_FOLDER
from tests.utils import assert_image_similar_to_expectation


@pytest.mark.skipif(
    get_device() == "cpu", reason="TypeError: Got unsupported ScalarType BFloat16"
)
def test_fix_faces(filename_base_for_orig_outputs, filename_base_for_outputs):
    distorted_img = Image.open(f"{TESTS_FOLDER}/data/distorted_face.png")
    seed_everything(1)
    img = enhance_faces(distorted_img)

    distorted_img.save(f"{filename_base_for_orig_outputs}__orig.jpg")
    img_path = f"{filename_base_for_outputs}.png"
    assert_image_similar_to_expectation(img, img_path=img_path, threshold=2800)
