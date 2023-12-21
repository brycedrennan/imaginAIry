import pytest
from PIL import Image

from imaginairy.api import imagine
from imaginairy.enhancers.clip_masking import get_img_mask
from imaginairy.schema import ImaginePrompt
from imaginairy.utils import get_device
from tests import TESTS_FOLDER
from tests.utils import assert_image_similar_to_expectation


@pytest.mark.skipif(get_device() == "cpu", reason="Too slow to run on CPU")
def test_clip_masking(filename_base_for_outputs):
    img = Image.open(f"{TESTS_FOLDER}/data/girl_with_a_pearl_earring_large.jpg")

    for mask_modifier in ["*0.5", "*6", "+1", "+11", "+101", "-25"]:
        pred_bin, pred_grayscale = get_img_mask(
            img,
            f"face AND NOT (bandana OR hair OR blue fabric){{{mask_modifier}}}",
            threshold=0.5,
        )

        mask_modifier = mask_modifier.replace("*", "x")
        img_path = f"{filename_base_for_outputs}_mask{mask_modifier}_g.png"

        assert_image_similar_to_expectation(
            pred_grayscale, img_path=img_path, threshold=300
        )

        img_path = f"{filename_base_for_outputs}_mask{mask_modifier}_bin.png"
        assert_image_similar_to_expectation(pred_bin, img_path=img_path, threshold=10)

    prompt = ImaginePrompt(
        "woman in sparkly gold jacket",
        init_image=img,
        init_image_strength=0.5,
        # lower steps for faster tests
        steps=40,
        mask_prompt="(head OR face){*5}",
        mask_mode="keep",
        upscale=False,
        fix_faces=True,
        seed=42,
        # solver_type="plms",
    )

    result = next(imagine(prompt))
    img_path = f"{filename_base_for_outputs}.png"
    assert_image_similar_to_expectation(result.img, img_path=img_path, threshold=7000)
