"""Test fooocus inpaint patch integration."""

import pytest


@pytest.mark.gpu
def test_fooocus_inpaint_generates_image():
    """Full integration test: generate an inpainted image using fooocus patch."""
    from PIL import Image, ImageDraw

    from imaginairy.api.generate import imagine
    from imaginairy.schema import ImaginePrompt

    img = Image.new("RGB", (1024, 1024), color=(50, 100, 200))
    draw = ImageDraw.Draw(img)
    draw.rectangle([300, 300, 700, 700], fill=(200, 50, 50))

    mask = Image.new("L", (1024, 1024), color=0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle([300, 300, 700, 700], fill=255)

    prompt = ImaginePrompt(
        prompt="a beautiful garden with flowers",
        init_image=img,
        mask_image=mask,
        inpaint_method="patch",
        model_weights="sdxl",
        steps=30,
        seed=42,
        size=1024,
    )

    results = list(imagine(prompts=[prompt]))
    assert len(results) == 1
    assert results[0].img.size == (1024, 1024)
