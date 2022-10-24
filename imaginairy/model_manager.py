import gc
import logging
import os

import torch
from omegaconf import OmegaConf
from transformers import cached_path

from imaginairy.paths import PKG_ROOT
from imaginairy.utils import get_device, instantiate_from_config

logger = logging.getLogger(__name__)

MODEL_SHORTCUTS = {
    "SD-1.4": (
        "configs/stable-diffusion-v1.yaml",
        "https://huggingface.co/bstddev/sd-v1-4/resolve/main/sd-v1-4.ckpt",
    ),
    "SD-1.5": (
        "configs/stable-diffusion-v1.yaml",
        "https://huggingface.co/acheong08/SD-V1-5-cloned/resolve/main/v1-5-pruned-emaonly.ckpt",
    ),
}
DEFAULT_MODEL = "SD-1.5"

LOADED_MODELS = {}


class MemoryAwareModel:
    """Wraps a model to allow dynamic loading/unloading as needed"""

    def __init__(self, config_path, weights_path, half_mode=None):
        self._config_path = config_path
        self._weights_path = weights_path
        self._half_mode = half_mode
        self._model = None

        LOADED_MODELS[(self._config_path, self._weights_path)] = self

    def __getattr__(self, key):
        if key == "_model":
            #  http://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
            raise AttributeError()

        if self._model is None:
            # unload all models in LOADED_MODELS
            for model in LOADED_MODELS.values():
                model.unload_model()

            model = load_model_from_config(
                config=OmegaConf.load(f"{PKG_ROOT}/{self._config_path}"),
                weights_location=self._weights_path,
            )

            # only run half-mode on cuda. run it by default
            half_mode = self._half_mode is None and get_device() == "cuda"
            if half_mode:
                model = model.half()
            self._model = model

        return getattr(self._model, key)

    def unload_model(self):
        del self._model
        self._model = None
        gc.collect()


def load_model_from_config(config, weights_location):
    if weights_location.startswith("http"):
        ckpt_path = cached_path(weights_location)
    else:
        ckpt_path = weights_location
    logger.info(f"Loading model {ckpt_path} onto {get_device()} backend...")
    pl_sd = None
    try:
        pl_sd = torch.load(ckpt_path, map_location="cpu")
    except RuntimeError as e:
        if "PytorchStreamReader failed reading zip archive" in str(e):
            if weights_location.startswith("http"):
                logger.warning("Corrupt checkpoint. deleting and re-downloading...")
                os.remove(ckpt_path)
                ckpt_path = cached_path(weights_location)
                pl_sd = torch.load(ckpt_path, map_location="cpu")
        if pl_sd is None:
            raise e
    if "global_step" in pl_sd:
        logger.debug(f"Global Step: {pl_sd['global_step']}")
    state_dict = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if len(missing_keys) > 0:
        logger.debug(f"missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        logger.debug(f"unexpected keys: {unexpected_keys}")

    model.to(get_device())
    model.eval()
    return model


def get_diffusion_model(
    weights_location=DEFAULT_MODEL,
    config_path="configs/stable-diffusion-v1.yaml",
    half_mode=None,
):
    """
    Load a diffusion model

    Weights location may also be shortcut name, e.g. "SD-1.5"
    """
    if weights_location is None:
        weights_location = DEFAULT_MODEL

    if weights_location in MODEL_SHORTCUTS:
        config_path, weights_location = MODEL_SHORTCUTS[weights_location]

    key = (config_path, weights_location)
    if key not in LOADED_MODELS:
        MemoryAwareModel(
            config_path=config_path, weights_path=weights_location, half_mode=half_mode
        )

    model = LOADED_MODELS[key]
    # calling model attribute forces it to load
    model.num_timesteps_cond  # noqa
    return model
