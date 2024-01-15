import pytest

from imaginairy import config
from imaginairy.utils.downloads import parse_diffusers_repo_url
from imaginairy.utils.model_manager import (
    resolve_model_weights_config,
)


def test_resolved_paths():
    """Test that the resolved model path is correct."""
    model_weights_config = resolve_model_weights_config(config.DEFAULT_MODEL_WEIGHTS)
    assert config.DEFAULT_MODEL_WEIGHTS.lower() in model_weights_config.aliases

    model_weights_config = resolve_model_weights_config(
        model_weights="foo.ckpt",
        default_model_architecture="sd15",
    )
    assert model_weights_config.aliases == []
    assert "sd15" in model_weights_config.architecture.aliases

    model_weights_config = resolve_model_weights_config(
        model_weights="foo.ckpt", default_model_architecture="sd15", for_inpainting=True
    )
    assert model_weights_config.aliases == []
    assert "sd15-inpaint" in model_weights_config.architecture.aliases


hf_urls_cases = [
    ("", {}),
    (
        "https://huggingface.co/prompthero/zoom-v3/",
        {"author": "prompthero", "repo": "zoom-v3", "ref": None},
    ),
    (
        "https://huggingface.co/prompthero/zoom-v3",
        {"author": "prompthero", "repo": "zoom-v3", "ref": None},
    ),
    (
        "https://huggingface.co/prompthero/zoom-v3/tree/main",
        {"author": "prompthero", "repo": "zoom-v3", "ref": "main"},
    ),
    (
        "https://huggingface.co/prompthero/zoom-v3/tree/main/",
        {"author": "prompthero", "repo": "zoom-v3", "ref": "main"},
    ),
    (
        "https://huggingface.co/prompthero/zoom-v3/tree/6027e2fe2343bf0ed09a5883e027506950f182ed/",
        {
            "author": "prompthero",
            "repo": "zoom-v3",
            "ref": "6027e2fe2343bf0ed09a5883e027506950f182ed",
        },
    ),
]


@pytest.mark.parametrize(("url", "expected"), hf_urls_cases)
def test_parse_diffusers_repo_url(url, expected):
    result = parse_diffusers_repo_url(url)
    assert result == expected
