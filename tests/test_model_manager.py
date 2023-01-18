from imaginairy import config
from imaginairy.model_manager import resolve_model_paths


def test_resolved_paths():
    """Test that the resolved model path is correct."""
    model_metadata, weights_path, config_path = resolve_model_paths()
    assert model_metadata.short_name == config.DEFAULT_MODEL
    assert model_metadata.config_path == config_path
    default_config_path = config_path

    model_metadata, weights_path, config_path = resolve_model_paths(
        weights_path="foo.ckpt"
    )
    assert weights_path == "foo.ckpt"
    assert config_path == default_config_path
