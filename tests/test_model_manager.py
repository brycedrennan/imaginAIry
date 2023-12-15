from imaginairy import config
from imaginairy.utils.model_manager import resolve_model_weights_config


def test_resolved_paths():
    """Test that the resolved model path is correct."""
    model_weights_config = resolve_model_weights_config(config.DEFAULT_MODEL_WEIGHTS)
    assert config.DEFAULT_MODEL_WEIGHTS.lower() in model_weights_config.aliases
    assert (
        config.DEFAULT_MODEL_ARCHITECTURE in model_weights_config.architecture.aliases
    )

    model_weights_config = resolve_model_weights_config(
        model_weights="foo.ckpt",
        default_model_architecture="sd15",
    )
    print(model_weights_config)
    assert model_weights_config.aliases == []
    assert "sd15" in model_weights_config.architecture.aliases

    model_weights_config = resolve_model_weights_config(
        model_weights="foo.ckpt", default_model_architecture="sd15", for_inpainting=True
    )
    assert model_weights_config.aliases == []
    assert "sd15-inpaint" in model_weights_config.architecture.aliases
