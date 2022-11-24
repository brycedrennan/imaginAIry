import gc
import glob
import logging
import os

import requests
import torch
from omegaconf import OmegaConf
from transformers import cached_path
from transformers.utils.hub import TRANSFORMERS_CACHE, HfFolder
from transformers.utils.hub import url_to_filename as tf_url_to_filename

from imaginairy.paths import PKG_ROOT
from imaginairy.utils import get_device, instantiate_from_config

logger = logging.getLogger(__name__)

MODEL_SHORTCUTS = {
    "SD-1.4": (
        "configs/stable-diffusion-v1.yaml",
        "https://huggingface.co/bstddev/sd-v1-4/resolve/77221977fa8de8ab8f36fac0374c120bd5b53287/sd-v1-4.ckpt",
    ),
    "SD-1.5": (
        "configs/stable-diffusion-v1.yaml",
        "https://huggingface.co/acheong08/SD-V1-5-cloned/resolve/fc392f6bd4345b80fc2256fa8aded8766b6c629e/v1-5-pruned-emaonly.ckpt",
    ),
    "SD-1.5-inpaint": (
        "configs/stable-diffusion-v1-inpaint.yaml",
        "https://huggingface.co/julienacquaviva/inpainting/resolve/2155ff7fe38b55f4c0d99c2f1ab9b561f8311ca7/sd-v1-5-inpainting.ckpt",
    ),
    "SD-2.0": (
        "configs/stable-diffusion-v2-inference.yaml",
        "https://huggingface.co/stabilityai/stable-diffusion-2-base/resolve/main/512-base-ema.ckpt",
    ),
    "SD-2.0-inpaint": (
        "configs/stable-diffusion-v2-inpainting-inference.yaml",
        "https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/resolve/main/512-inpainting-ema.ckpt",
    ),
    "SD-2.0-v": (
        "configs/stable-diffusion-v2-inference-v.yaml",
        "https://huggingface.co/stabilityai/stable-diffusion-2/resolve/main/768-v-ema.ckpt",
    ),
}
DEFAULT_MODEL = "SD-2.0"

LOADED_MODELS = {}
MOST_RECENTLY_LOADED_MODEL = None


class HuggingFaceAuthorizationError(RuntimeError):
    pass


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
        ckpt_path = get_cached_url_path(weights_location)
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
                ckpt_path = get_cached_url_path(weights_location)
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
    for_inpainting=False,
):
    """
    Load a diffusion model

    Weights location may also be shortcut name, e.g. "SD-1.5"
    """
    try:
        return _get_diffusion_model(
            weights_location, config_path, half_mode, for_inpainting
        )
    except HuggingFaceAuthorizationError as e:
        if for_inpainting:
            logger.warning(
                f"Failed to load inpainting model. Attempting to fall-back to standard model.   {str(e)}"
            )
            return _get_diffusion_model(
                DEFAULT_MODEL, config_path, half_mode, for_inpainting=False
            )
        raise e


def _get_diffusion_model(
    weights_location=DEFAULT_MODEL,
    config_path="configs/stable-diffusion-v1.yaml",
    half_mode=None,
    for_inpainting=False,
):
    """
    Load a diffusion model

    Weights location may also be shortcut name, e.g. "SD-1.5"
    """
    global MOST_RECENTLY_LOADED_MODEL  # noqa
    if weights_location is None:
        weights_location = DEFAULT_MODEL
    if for_inpainting and f"{weights_location}-inpaint" in MODEL_SHORTCUTS:
        config_path, weights_location = MODEL_SHORTCUTS[f"{weights_location}-inpaint"]
    elif weights_location in MODEL_SHORTCUTS:
        config_path, weights_location = MODEL_SHORTCUTS[weights_location]

    key = (config_path, weights_location)
    if key not in LOADED_MODELS:
        MemoryAwareModel(
            config_path=config_path, weights_path=weights_location, half_mode=half_mode
        )

    model = LOADED_MODELS[key]
    # calling model attribute forces it to load
    model.num_timesteps_cond  # noqa
    MOST_RECENTLY_LOADED_MODEL = model
    return model


def get_current_diffusion_model():
    return MOST_RECENTLY_LOADED_MODEL


def get_cache_dir():
    xdg_cache_home = os.getenv("XDG_CACHE_HOME", None)
    if xdg_cache_home is None:
        user_home = os.getenv("HOME", None)
        if user_home:
            xdg_cache_home = os.path.join(user_home, ".cache")

    if xdg_cache_home is not None:
        return os.path.join(xdg_cache_home, "imaginairy", "weights")

    return os.path.join(os.path.dirname(__file__), ".cached-downloads")


def get_cached_url_path(url):
    """
    Gets the contents of a url, but caches the response indefinitely

    While we attempt to use the cached_path from huggingface transformers, we fall back
    to our own implementation if the url does not provide an etag header, which `cached_path`
    requires.  We also skip the `head` call that `cached_path` makes on every call if the file
    is already cached.
    """

    try:
        return huggingface_cached_path(url)
    except (OSError, ValueError):
        pass
    filename = url.split("/")[-1]
    dest = get_cache_dir()
    os.makedirs(dest, exist_ok=True)
    dest_path = os.path.join(dest, filename)
    if os.path.exists(dest_path):
        return dest_path
    r = requests.get(url)  # noqa

    with open(dest_path, "wb") as f:
        f.write(r.content)
    return dest_path


def find_url_in_huggingface_cache(url):
    huggingface_filename = os.path.join(TRANSFORMERS_CACHE, tf_url_to_filename(url))
    for name in glob.glob(huggingface_filename + "*"):
        if name.endswith((".json", ".lock")):
            continue

        return name
    return None


def check_huggingface_url_authorized(url):
    if not url.startswith("https://huggingface.co/"):
        return None
    token = HfFolder.get_token()
    headers = {}
    if token is not None:
        headers["authorization"] = f"Bearer {token}"
    response = requests.head(url, allow_redirects=True, headers=headers, timeout=5)
    if response.status_code == 401:
        raise HuggingFaceAuthorizationError(
            "Unauthorized access to HuggingFace model. This model requires a huggingface token.  "
            "Please login to HuggingFace "
            "or set HUGGING_FACE_HUB_TOKEN to your User Access Token. "
            "See https://huggingface.co/docs/huggingface_hub/quick-start#login for more information"
        )
    return None


def huggingface_cached_path(url):
    # bypass all the HEAD calls done by the default `cached_path`
    dest_path = find_url_in_huggingface_cache(url)
    if not dest_path:
        check_huggingface_url_authorized(url)
        token = HfFolder.get_token()
        dest_path = cached_path(url, use_auth_token=token)
    return dest_path
