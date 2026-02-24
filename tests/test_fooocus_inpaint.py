"""Test fooocus inpaint patch integration."""

import pytest


@pytest.mark.gpu
def test_fooocus_inpaint_generates_image():
    """Full integration test: generate an inpainted image using fooocus patch.

    Uses a large image with a small mask to exercise ROI cropping.
    """
    from PIL import Image, ImageDraw

    from imaginairy.api.generate import imagine
    from imaginairy.schema import ImaginePrompt

    # Large image with a small mask region triggers ROI crop + upscale
    img = Image.new("RGB", (1536, 1024), color=(50, 100, 200))
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 400, 400], fill=(200, 50, 50))

    mask = Image.new("L", (1536, 1024), color=0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle([100, 100, 400, 400], fill=255)

    prompt = ImaginePrompt(
        prompt="a beautiful garden with flowers",
        init_image=img,
        mask_image=mask,
        model_weights="sdxl",
        inpaint_method="patch",
        steps=30,
        seed=42,
        size="1536x1024",
    )

    results = list(imagine(prompts=[prompt]))
    assert len(results) == 1
    assert results[0].img.size == (1536, 1024)


def test_fooocus_fill():
    """Test that fooocus_fill fills masked region with blurred surroundings."""
    from PIL import Image

    from imaginairy.modules.fooocus_inpaint import fooocus_fill

    img = Image.new("RGB", (256, 256), color=(100, 150, 200))
    mask = Image.new("L", (256, 256), color=0)
    # Mask the center
    for y in range(64, 192):
        for x in range(64, 192):
            mask.putpixel((x, y), 255)

    result = fooocus_fill(img, mask)
    assert result.size == (256, 256)
    # Center pixel should be close to the surrounding color, not black
    center = result.getpixel((128, 128))
    assert center[0] > 50  # not black


def test_morphological_open_mask():
    """Test that morphological_open_mask dilates and blurs."""
    import numpy as np
    from PIL import Image

    from imaginairy.modules.fooocus_inpaint import morphological_open_mask

    mask = Image.new("L", (256, 256), color=0)
    # Small white square in center
    for y in range(112, 144):
        for x in range(112, 144):
            mask.putpixel((x, y), 255)

    result = morphological_open_mask(mask)
    result_np = np.array(result)

    # Result should have values between 0 and 255 (soft edges from dilate-8)
    assert result_np.max() > 200
    # Edge region should have intermediate values (gradual falloff)
    assert 0 < result_np[100, 128] < 255


def test_prepare_inpaint_crop_small_mask():
    """Test that prepare_inpaint_crop crops and resizes to ~1024."""
    from PIL import Image

    from imaginairy.modules.fooocus_inpaint import prepare_inpaint_crop

    img = Image.new("RGB", (2048, 2048), color=(100, 100, 100))
    mask = Image.new("L", (2048, 2048), color=0)
    # Small mask in corner
    for y in range(100, 200):
        for x in range(100, 200):
            mask.putpixel((x, y), 255)

    cropped_img, cropped_mask, crop = prepare_inpaint_crop(img, mask)
    # Working size should be near 1024
    w, h = cropped_img.size
    assert abs((w * h) ** 0.5 - 1024) < 128
    assert cropped_mask.size == cropped_img.size
    # Crop box should be within image bounds
    y1, y2, x1, x2 = crop.crop_box
    assert 0 <= y1 < y2 <= 2048
    assert 0 <= x1 < x2 <= 2048


def test_prepare_inpaint_crop_full_mask():
    """Test that prepare_inpaint_crop still works with a full mask."""
    from PIL import Image

    from imaginairy.modules.fooocus_inpaint import prepare_inpaint_crop

    img = Image.new("RGB", (1024, 1024), color=(100, 100, 100))
    mask = Image.new("L", (1024, 1024), color=255)  # full mask

    cropped_img, cropped_mask, _crop = prepare_inpaint_crop(img, mask)
    # Should still return a valid crop (uses whole image)
    assert cropped_img.size[0] > 0
    assert cropped_mask.size == cropped_img.size
