import gc
import logging
import os
import re
import sys
import urllib.parse
from functools import wraps

import requests
import torch
from huggingface_hub import hf_hub_download as _hf_hub_download
from huggingface_hub import try_to_load_from_cache
from omegaconf import OmegaConf
from safetensors.torch import load_file
from transformers.utils.hub import HfFolder

from imaginairy import config as iconfig
from imaginairy.config import MODEL_SHORT_NAMES
from imaginairy.modules import attention
from imaginairy.paths import PKG_ROOT
from imaginairy.utils import get_device, instantiate_from_config

logger = logging.getLogger(__name__)


LOADED_MODELS = {}
MOST_RECENTLY_LOADED_MODEL = None


class HuggingFaceAuthorizationError(RuntimeError):
    pass


class MemoryAwareModel:
    """Wraps a model to allow dynamic loading/unloading as needed."""

    def __init__(self, config_path, weights_path, half_mode=None, for_training=False):
        self._config_path = config_path
        self._weights_path = weights_path
        self._half_mode = half_mode
        self._model = None
        self._for_training = for_training

        LOADED_MODELS[(self._config_path, self._weights_path)] = self

    def __getattr__(self, key):
        if key == "_model":
            #  http://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
            raise AttributeError()

        if self._model is None:
            # unload all models in LOADED_MODELS
            for model in LOADED_MODELS.values():
                model.unload_model()
            model_config = OmegaConf.load(f"{PKG_ROOT}/{self._config_path}")
            if self._for_training:
                model_config.use_ema = True
                # model_config.use_scheduler = True

            model = load_model_from_config(
                config=model_config,
                weights_location=self._weights_path,
            )

            # only run half-mode on cuda. run it by default
            half_mode = self._half_mode is None and get_device() == "cuda"
            if half_mode:
                model = model.half()
            self._model = model

        return getattr(self._model, key)

    def unload_model(self):
        if self._model is not None:
            del self._model.cond_stage_model
            del self._model.first_stage_model
            del self._model.model
            del self._model
            self._model = None
        if get_device() == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


def load_tensors(tensorfile, map_location=None):
    if tensorfile.endswith(".ckpt"):
        return torch.load(tensorfile, map_location=map_location)
    if tensorfile.endswith(".safetensors"):
        return load_file(tensorfile, device=map_location)
    raise ValueError(f"Unknown tensorfile type: {tensorfile}")


def load_model_from_config(config, weights_location):
    if weights_location.startswith("http"):
        ckpt_path = get_cached_url_path(weights_location, category="weights")
    else:
        ckpt_path = weights_location
    logger.info(f"Loading model {ckpt_path} onto {get_device()} backend...")
    pl_sd = None
    try:
        pl_sd = load_tensors(ckpt_path, map_location="cpu")
    except FileNotFoundError as e:
        if e.errno == 2:
            logger.error(
                f'Error: "{ckpt_path}" not a valid path to model weights.\nPreconfigured models you can use: {MODEL_SHORT_NAMES}.'
            )
            sys.exit(1)
        raise e
    except RuntimeError as e:
        if "PytorchStreamReader failed reading zip archive" in str(e):
            if weights_location.startswith("http"):
                logger.warning("Corrupt checkpoint. deleting and re-downloading...")
                os.remove(ckpt_path)
                ckpt_path = get_cached_url_path(weights_location, category="weights")
                pl_sd = load_tensors(ckpt_path, map_location="cpu")
        if pl_sd is None:
            raise e
    if "global_step" in pl_sd:
        logger.debug(f"Global Step: {pl_sd['global_step']}")
    if "state_dict" in pl_sd:
        state_dict = pl_sd["state_dict"]
    else:
        state_dict = pl_sd
    model = instantiate_from_config(config.model)
    model.init_from_state_dict(state_dict)

    model.to(get_device())
    model.eval()
    return model


def get_diffusion_model(
    weights_location=iconfig.DEFAULT_MODEL,
    config_path="configs/stable-diffusion-v1.yaml",
    half_mode=None,
    for_inpainting=False,
    for_training=False,
):
    """
    Load a diffusion model.

    Weights location may also be shortcut name, e.g. "SD-1.5"
    """
    try:
        return _get_diffusion_model(
            weights_location,
            config_path,
            half_mode,
            for_inpainting,
            for_training=for_training,
        )
    except HuggingFaceAuthorizationError as e:
        if for_inpainting:
            logger.warning(
                f"Failed to load inpainting model. Attempting to fall-back to standard model.   {str(e)}"
            )
            return _get_diffusion_model(
                iconfig.DEFAULT_MODEL,
                config_path,
                half_mode,
                for_inpainting=False,
                for_training=for_training,
            )
        raise e


def _get_diffusion_model(
    weights_location=iconfig.DEFAULT_MODEL,
    config_path="configs/stable-diffusion-v1.yaml",
    half_mode=None,
    for_inpainting=False,
    for_training=False,
):
    """
    Load a diffusion model.

    Weights location may also be shortcut name, e.g. "SD-1.5"
    """
    global MOST_RECENTLY_LOADED_MODEL  # noqa
    model_config, weights_location, config_path = resolve_model_paths(
        weights_path=weights_location,
        config_path=config_path,
        for_inpainting=for_inpainting,
        for_training=for_training,
    )
    # some models need the attention calculated in float32
    if model_config is not None:
        attention.ATTENTION_PRECISION_OVERRIDE = model_config.forced_attn_precision
    else:
        attention.ATTENTION_PRECISION_OVERRIDE = "default"

    key = (config_path, weights_location)
    if key not in LOADED_MODELS:
        MemoryAwareModel(
            config_path=config_path,
            weights_path=weights_location,
            half_mode=half_mode,
            for_training=for_training,
        )

    model = LOADED_MODELS[key]
    # calling model attribute forces it to load
    model.num_timesteps_cond  # noqa
    MOST_RECENTLY_LOADED_MODEL = model
    return model


def resolve_model_paths(
    weights_path=iconfig.DEFAULT_MODEL,
    config_path=None,
    for_inpainting=False,
    for_training=False,
):
    """Resolve weight and config path if they happen to be shortcuts."""
    model_metadata_w = iconfig.MODEL_CONFIG_SHORTCUTS.get(weights_path, None)
    model_metadata_c = iconfig.MODEL_CONFIG_SHORTCUTS.get(config_path, None)
    if for_inpainting:
        model_metadata_w = iconfig.MODEL_CONFIG_SHORTCUTS.get(
            f"{weights_path}-inpaint", model_metadata_w
        )
        model_metadata_c = iconfig.MODEL_CONFIG_SHORTCUTS.get(
            f"{config_path}-inpaint", model_metadata_c
        )

    if model_metadata_w:
        if config_path is None:
            config_path = model_metadata_w.config_path
        if for_training:
            weights_path = model_metadata_w.weights_url_full
            if weights_path is None:
                raise ValueError(
                    "No full training weights configured for this model. Edit the code or subimt a github issue."
                )
        else:
            weights_path = model_metadata_w.weights_url

    if model_metadata_c:
        config_path = model_metadata_c.config_path

    if config_path is None:
        config_path = iconfig.MODEL_CONFIG_SHORTCUTS[iconfig.DEFAULT_MODEL].config_path
    model_metadata = model_metadata_w or model_metadata_c
    logger.debug(f"Loading model weights from: {weights_path}")
    logger.debug(f"Loading model config from:  {config_path}")
    return model_metadata, weights_path, config_path


def get_model_default_image_size(weights_location):
    model_config = iconfig.MODEL_CONFIG_SHORTCUTS.get(weights_location, None)
    if model_config:
        return model_config.default_image_size
    return 512


def get_current_diffusion_model():
    return MOST_RECENTLY_LOADED_MODEL


def get_cache_dir():
    xdg_cache_home = os.getenv("XDG_CACHE_HOME", None)
    if xdg_cache_home is None:
        user_home = os.getenv("HOME", None)
        if user_home:
            xdg_cache_home = os.path.join(user_home, ".cache")

    if xdg_cache_home is not None:
        return os.path.join(xdg_cache_home, "imaginairy")

    return os.path.join(os.path.dirname(__file__), ".cached-aimg")


def get_cached_url_path(url, category=None):
    """
    Gets the contents of a url, but caches the response indefinitely.

    While we attempt to use the cached_path from huggingface transformers, we fall back
    to our own implementation if the url does not provide an etag header, which `cached_path`
    requires.  We also skip the `head` call that `cached_path` makes on every call if the file
    is already cached.
    """

    try:
        if url.startswith("https://huggingface.co"):
            return huggingface_cached_path(url)
    except (OSError, ValueError):
        pass
    filename = url.split("/")[-1]
    dest = get_cache_dir()
    if category:
        dest = os.path.join(dest, category)
    os.makedirs(dest, exist_ok=True)

    # Replace possibly illegal destination path characters
    safe_filename = re.sub('[*<>:"|?]', "_", filename)
    dest_path = os.path.join(dest, safe_filename)
    if os.path.exists(dest_path):
        return dest_path

    # check if it's saved at previous path and rename it
    old_dest_path = os.path.join(dest, filename)
    if os.path.exists(old_dest_path):
        os.rename(old_dest_path, dest_path)
        return dest_path

    r = requests.get(url)  # noqa

    with open(dest_path, "wb") as f:
        f.write(r.content)
    return dest_path


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


@wraps(_hf_hub_download)
def hf_hub_download(*args, **kwargs):
    """
    backwards compatible wrapper for huggingface's hf_hub_download.

    they changed the argument name from `use_auth_token` to `token`
    """

    try:
        return _hf_hub_download(*args, **kwargs)
    except TypeError as e:
        if "unexpected keyword argument 'token'" in str(e):
            kwargs["use_auth_token"] = kwargs.pop("token")
            return _hf_hub_download(*args, **kwargs)
        raise e


def huggingface_cached_path(url):
    # bypass all the HEAD calls done by the default `cached_path`
    repo, commit_hash, filepath = extract_huggingface_repo_commit_file_from_url(url)
    dest_path = try_to_load_from_cache(
        repo_id=repo, revision=commit_hash, filename=filepath
    )
    if not dest_path:
        check_huggingface_url_authorized(url)
        token = HfFolder.get_token()
        logger.info(f"Downloading {url} from huggingface")
        dest_path = hf_hub_download(
            repo_id=repo, revision=commit_hash, filename=filepath, token=token
        )
        # make a refs folder so caching works
        # work-around for
        # https://github.com/huggingface/huggingface_hub/pull/1306
        # https://github.com/brycedrennan/imaginAIry/issues/171
        refs_url = dest_path[: dest_path.index("/snapshots/")] + "/refs/"
        os.makedirs(refs_url, exist_ok=True)
    return dest_path


def extract_huggingface_repo_commit_file_from_url(url):
    parsed_url = urllib.parse.urlparse(url)
    path_components = parsed_url.path.strip("/").split("/")

    repo = "/".join(path_components[0:2])
    assert path_components[2] == "resolve"
    commit_hash = path_components[3]
    filepath = "/".join(path_components[4:])

    return repo, commit_hash, filepath
