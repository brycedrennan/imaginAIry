import hashlib

from PIL import Image
from pytorch_lightning import seed_everything

from imaginairy.enhancers.face_restoration_codeformer import enhance_faces
from imaginairy.utils import get_device
from tests import TESTS_FOLDER


def test_fix_faces():
    img = Image.open(f"{TESTS_FOLDER}/data/distorted_face.png")
    seed_everything(1)
    img = enhance_faces(img)
    img.save(f"{TESTS_FOLDER}/test_output/fixed_face.png")
    if "mps" in get_device():
        assert img_hash(img) == "a75991307eda675a26eeb7073f828e93"
    else:
        assert img_hash(img) == "5aa847a1464de75b158658a35800b6bf"


def img_hash(img):
    return hashlib.md5(img.tobytes()).hexdigest()
