import pytest

from imaginairy.utils.named_resolutions import normalize_image_size

valid_cases = [
    ("HD", (1280, 720)),
    ("FHD", (1920, 1080)),
    ("hd", (1280, 720)),
    ("fhd", (1920, 1080)),
    ("Hd", (1280, 720)),
    ("FhD", (1920, 1080)),
    ("1920x1080", (1920, 1080)),
    ("1280x720", (1280, 720)),
    ("1024x768", (1024, 768)),
    ("1024,768", (1024, 768)),
    ("1024*768", (1024, 768)),
    ("1024, 768", (1024, 768)),
    ("800", (800, 800)),
    ("1024", (1024, 1024)),
    ("1080p", (1920, 1080)),
    ("1080P", (1920, 1080)),
    (512, (512, 512)),
    ((512, 512), (512, 512)),
    ("1x1", (1, 1)),
]
invalid_cases = [
    None,
    3.14,
    (3.14, 3.14),
    "",
    "     ",
    "abc",
    "-512",
    "1920xABC",
    "1920x1080x1234",
    "x1920",
    "123.1",
    "12x",
    "x12",
    "x",
    "12x12x12x12",
]


@pytest.mark.parametrize(("named_resolution", "expected"), valid_cases)
def test_named_resolutions(named_resolution, expected):
    assert normalize_image_size(named_resolution) == expected


@pytest.mark.parametrize("named_resolution", invalid_cases)
def test_invalid_inputs(named_resolution):
    with pytest.raises(ValueError, match="Invalid resolution"):
        normalize_image_size(named_resolution)
