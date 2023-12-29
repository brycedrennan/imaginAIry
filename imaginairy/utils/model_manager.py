"""Classes and functions for managing AI models"""

import logging
import os
import re
import sys
import urllib.parse
from functools import lru_cache, wraps

import requests
import torch
from huggingface_hub import (
    HfFileSystem,
    HfFolder,
    hf_hub_download as _hf_hub_download,
    try_to_load_from_cache,
)
from omegaconf import OmegaConf
from refiners.foundationals.latent_diffusion import DoubleTextEncoder, SD1UNet, SDXLUNet
from refiners.foundationals.latent_diffusion.model import LatentDiffusionModel
from safetensors.torch import load_file

from imaginairy import config as iconfig
from imaginairy.config import IMAGE_WEIGHTS_SHORT_NAMES, ModelArchitecture
from imaginairy.modules import attention
from imaginairy.modules.refiners_sd import SDXLAutoencoderSliced, StableDiffusion_XL
from imaginairy.utils import clear_gpu_cache, get_device, instantiate_from_config
from imaginairy.utils.model_cache import memory_managed_model
from imaginairy.utils.named_resolutions import normalize_image_size
from imaginairy.utils.paths import PKG_ROOT
from imaginairy.weight_management import translators

logger = logging.getLogger(__name__)

MOST_RECENTLY_LOADED_MODEL = None


class HuggingFaceAuthorizationError(RuntimeError):
    pass


def load_state_dict(weights_location, half_mode=False, device=None):
    if device is None:
        device = get_device()

    if weights_location.startswith("http"):
        ckpt_path = get_cached_url_path(weights_location, category="weights")
    else:
        ckpt_path = weights_location
    logger.info(f"Loading model {ckpt_path} onto {get_device()} backend...")
    state_dict = None
    # weights_cache_key = (ckpt_path, half_mode)
    # if weights_cache_key in GLOBAL_WEIGHTS_CACHE:
    #     return GLOBAL_WEIGHTS_CACHE.get(weights_cache_key)

    try:
        state_dict = load_tensors(ckpt_path, map_location="cpu")
    except FileNotFoundError as e:
        if e.errno == 2:
            logger.error(
                f'Error: "{ckpt_path}" not a valid path to model weights.\nPreconfigured models you can use: {IMAGE_WEIGHTS_SHORT_NAMES}.'
            )
            sys.exit(1)
        raise
    except RuntimeError as e:
        err_str = str(e)
        if (
            "PytorchStreamReader failed reading zip archive" in err_str
            and weights_location.startswith("http")
        ):
            logger.warning("Corrupt checkpoint. deleting and re-downloading...")
            os.remove(ckpt_path)
            ckpt_path = get_cached_url_path(weights_location, category="weights")
            state_dict = load_tensors(ckpt_path, map_location="cpu")
        if state_dict is None:
            raise

    state_dict = state_dict.get("state_dict", state_dict)

    if half_mode:
        state_dict = {k: v.half() for k, v in state_dict.items()}

    # change device
    state_dict = {k: v.to(device) for k, v in state_dict.items()}

    # GLOBAL_WEIGHTS_CACHE.set(weights_cache_key, state_dict)

    return state_dict


def load_model_from_config(config, weights_location, half_mode=False):
    model = instantiate_from_config(config.model)
    base_model_dict = load_state_dict(weights_location, half_mode=half_mode)
    model.init_from_state_dict(base_model_dict)
    if half_mode:
        model = model.half()
    model.to(get_device())
    model.eval()
    return model


def load_model_from_config_old(
    config, weights_location, control_weights_locations=None, half_mode=False
):
    model = instantiate_from_config(config.model)
    base_model_dict = load_state_dict(weights_location, half_mode=half_mode)
    model.init_from_state_dict(base_model_dict)

    control_weights_locations = control_weights_locations or []
    controlnets = []
    for control_weights_location in control_weights_locations:
        controlnet_state_dict = load_state_dict(
            control_weights_location, half_mode=half_mode
        )
        controlnet_state_dict = {
            k.replace("control_model.", ""): v for k, v in controlnet_state_dict.items()
        }
        controlnet = instantiate_from_config(model.control_stage_config)
        controlnet.load_state_dict(controlnet_state_dict)
        controlnet.to(get_device())
        controlnets.append(controlnet)
        model.set_control_models(controlnets)

    if half_mode:
        model = model.half()
    print("halved")

    model.to(get_device())
    print("moved to device")
    model.eval()
    print("set to eval mode")
    return model


def add_controlnet(base_state_dict, controlnet_state_dict):
    """Merges a base sd15 model with a controlnet model."""
    for key in controlnet_state_dict:
        base_state_dict[key] = controlnet_state_dict[key]
    return base_state_dict


def get_diffusion_model(
    weights_location=iconfig.DEFAULT_MODEL_WEIGHTS,
    config_path="configs/stable-diffusion-v1.yaml",
    control_weights_locations=None,
    half_mode=None,
    for_inpainting=False,
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
            control_weights_locations=control_weights_locations,
        )
    except HuggingFaceAuthorizationError as e:
        if for_inpainting:
            logger.warning(
                f"Failed to load inpainting model. Attempting to fall-back to standard model.   {e!s}"
            )
            return _get_diffusion_model(
                iconfig.DEFAULT_MODEL_WEIGHTS,
                config_path,
                half_mode,
                for_inpainting=False,
                control_weights_locations=control_weights_locations,
            )
        raise


def _get_diffusion_model(
    weights_location=iconfig.DEFAULT_MODEL_WEIGHTS,
    model_architecture="configs/stable-diffusion-v1.yaml",
    half_mode=None,
    for_inpainting=False,
    control_weights_locations=None,
):
    """
    Load a diffusion model.

    Weights location may also be shortcut name, e.g. "SD-1.5"
    """
    global MOST_RECENTLY_LOADED_MODEL

    model_weights_config = resolve_model_weights_config(
        model_weights=weights_location,
        default_model_architecture=model_architecture,
        for_inpainting=for_inpainting,
    )
    # some models need the attention calculated in float32
    if model_weights_config is not None:
        attention.ATTENTION_PRECISION_OVERRIDE = (
            model_weights_config.forced_attn_precision
        )
    else:
        attention.ATTENTION_PRECISION_OVERRIDE = "default"
    diffusion_model = _load_diffusion_model(
        config_path=model_weights_config.architecture.config_path,
        weights_location=weights_location,
        half_mode=half_mode,
    )
    MOST_RECENTLY_LOADED_MODEL = diffusion_model
    if control_weights_locations:
        controlnets = []
        for control_weights_location in control_weights_locations:
            controlnets.append(load_controlnet(control_weights_location, half_mode))
        diffusion_model.set_control_models(controlnets)

    return diffusion_model


def get_diffusion_model_refiners(
    weights_config: iconfig.ModelWeightsConfig,
    for_inpainting=False,
    dtype=None,
) -> LatentDiffusionModel:
    """Load a diffusion model."""

    return _get_diffusion_model_refiners(
        weights_location=weights_config.weights_location,
        architecture_alias=weights_config.architecture.primary_alias,
        for_inpainting=for_inpainting,
        dtype=dtype,
    )


hf_repo_url_pattern = re.compile(
    r"https://huggingface\.co/(?P<author>[^/]+)/(?P<repo>[^/]+)(/tree/(?P<ref>[a-z0-9]+))?/?$"
)


def parse_diffusers_repo_url(url: str) -> dict[str, str]:
    match = hf_repo_url_pattern.match(url)
    return match.groupdict() if match else {}


def is_diffusers_repo_url(url: str) -> bool:
    return bool(parse_diffusers_repo_url(url))


def normalize_diffusers_repo_url(url: str) -> str:
    data = parse_diffusers_repo_url(url)
    ref = data["ref"] or "main"
    normalized_url = (
        f"https://huggingface.co/{data['author']}/{data['repo']}/tree/{ref}/"
    )
    return normalized_url


@lru_cache(maxsize=1)
def _get_diffusion_model_refiners(
    weights_location: str,
    architecture_alias: str,
    for_inpainting: bool = False,
    device=None,
    dtype=torch.float16,
) -> LatentDiffusionModel:
    """
    Load a diffusion model.

    Weights location may also be shortcut name, e.g. "SD-1.5"
    """
    global MOST_RECENTLY_LOADED_MODEL
    _get_diffusion_model_refiners.cache_clear()
    clear_gpu_cache()

    architecture = iconfig.MODEL_ARCHITECTURE_LOOKUP[architecture_alias]
    if architecture.primary_alias in ("sd15", "sd15inpaint"):
        sd = _get_sd15_diffusion_model_refiners(
            weights_location=weights_location,
            for_inpainting=for_inpainting,
            device=device,
            dtype=dtype,
        )
    elif architecture.primary_alias == "sdxl":
        sd = load_sdxl_pipeline(base_url=weights_location, device=device)
    else:
        msg = f"Invalid architecture {architecture.primary_alias}"
        raise ValueError(msg)

    MOST_RECENTLY_LOADED_MODEL = sd
    return sd


def _get_sd15_diffusion_model_refiners(
    weights_location: str,
    for_inpainting: bool = False,
    device=None,
    dtype=torch.float16,
) -> LatentDiffusionModel:
    """
    Load a diffusion model.

    Weights location may also be shortcut name, e.g. "SD-1.5"
    """
    from imaginairy.modules.refiners_sd import (
        SD1AutoencoderSliced,
        StableDiffusion_1,
        StableDiffusion_1_Inpainting,
    )

    device = device or get_device()
    if is_diffusers_repo_url(weights_location):
        (
            vae_weights,
            unet_weights,
            text_encoder_weights,
        ) = load_sd15_diffusers_weights(weights_location)
    else:
        (
            vae_weights,
            unet_weights,
            text_encoder_weights,
        ) = load_stable_diffusion_compvis_weights(weights_location)

    StableDiffusionCls: type[LatentDiffusionModel]
    if for_inpainting:
        unet = SD1UNet(in_channels=9, device=device, dtype=dtype)
        StableDiffusionCls = StableDiffusion_1_Inpainting
    else:
        unet = SD1UNet(in_channels=4, device=device, dtype=dtype)
        StableDiffusionCls = StableDiffusion_1
    logger.debug(f"Using class {StableDiffusionCls.__name__}")

    sd = StableDiffusionCls(
        device=device, dtype=dtype, lda=SD1AutoencoderSliced(), unet=unet
    )
    logger.debug("Loading VAE")
    sd.lda.load_state_dict(vae_weights)

    logger.debug("Loading text encoder")
    sd.clip_text_encoder.load_state_dict(text_encoder_weights)

    logger.debug("Loading UNet")
    sd.unet.load_state_dict(unet_weights, strict=False)

    logger.debug(f"'{weights_location}' Loaded")

    sd.set_self_attention_guidance(enable=True)

    return sd


@memory_managed_model("stable-diffusion", memory_usage_mb=1951)
def _load_diffusion_model(config_path, weights_location, half_mode):
    model_config = OmegaConf.load(f"{PKG_ROOT}/{config_path}")

    # only run half-mode on cuda. run it by default
    half_mode = half_mode is None and get_device() == "cuda"

    model = load_model_from_config(
        config=model_config,
        weights_location=weights_location,
        half_mode=half_mode,
    )
    return model


@memory_managed_model("controlnet")
def load_controlnet(control_weights_location, half_mode):
    controlnet_state_dict = load_state_dict(
        control_weights_location, half_mode=half_mode
    )
    controlnet_state_dict = {
        k.replace("control_model.", ""): v for k, v in controlnet_state_dict.items()
    }
    control_stage_config = OmegaConf.load(f"{PKG_ROOT}/configs/control-net-v15.yaml")[
        "model"
    ]["params"]["control_stage_config"]
    controlnet = instantiate_from_config(control_stage_config)
    controlnet.load_state_dict(controlnet_state_dict)
    controlnet.to(get_device())
    return controlnet


def resolve_model_weights_config(
    model_weights: str | iconfig.ModelWeightsConfig,
    default_model_architecture: str | None = None,
    for_inpainting: bool = False,
) -> iconfig.ModelWeightsConfig:
    """Resolve weight and config path if they happen to be shortcuts."""
    if isinstance(model_weights, iconfig.ModelWeightsConfig):
        return model_weights

    if not isinstance(model_weights, str):
        msg = f"Invalid model weights: {model_weights}"
        raise ValueError(msg)  # noqa

    if default_model_architecture is not None and not isinstance(
        default_model_architecture, str
    ):
        msg = f"Invalid model architecture: {default_model_architecture}"
        raise ValueError(msg)

    if for_inpainting:
        model_weights_config = iconfig.MODEL_WEIGHT_CONFIG_LOOKUP.get(
            f"{model_weights.lower()}-inpaint", None
        )
        if model_weights_config:
            return model_weights_config

    model_weights_config = iconfig.MODEL_WEIGHT_CONFIG_LOOKUP.get(
        model_weights.lower(), None
    )
    if model_weights_config:
        return model_weights_config

    if not default_model_architecture:
        msg = "You must specify the model architecture when loading custom weights."
        raise ValueError(msg)

    default_model_architecture = default_model_architecture.lower()
    model_architecture_config = None
    if for_inpainting:
        model_architecture_config = iconfig.MODEL_ARCHITECTURE_LOOKUP.get(
            f"{default_model_architecture}-inpaint", None
        )

    if not model_architecture_config:
        model_architecture_config = iconfig.MODEL_ARCHITECTURE_LOOKUP.get(
            default_model_architecture, None
        )

    if model_architecture_config is None:
        msg = f"Invalid model architecture: {default_model_architecture}"
        raise ValueError(msg)

    model_weights_config = iconfig.ModelWeightsConfig(
        name="Custom Loaded",
        aliases=[],
        architecture=model_architecture_config,
        weights_location=model_weights,
        defaults={},
    )

    return model_weights_config


def get_model_default_image_size(model_architecture: str | ModelArchitecture | None):
    if isinstance(model_architecture, str):
        model_architecture = iconfig.MODEL_ARCHITECTURE_LOOKUP.get(
            model_architecture, None
        )
    default_size = None
    if model_architecture:
        default_size = model_architecture.defaults.get("size")

    if default_size is None:
        default_size = 512
    default_size = normalize_image_size(default_size)
    return default_size


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

    r = requests.get(url)

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
        msg = "Unauthorized access to HuggingFace model. This model requires a huggingface token.  Please login to HuggingFace or set HUGGING_FACE_HUB_TOKEN to your User Access Token. See https://huggingface.co/docs/huggingface_hub/quick-start#login for more information"
        raise HuggingFaceAuthorizationError(msg)
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
        raise


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


def download_diffusers_weights(base_url, sub, filename=None, prefer_fp16=True):
    if filename is None:
        # select which weights to download. prefer fp16 safetensors
        data = parse_diffusers_repo_url(base_url)
        fs = HfFileSystem()
        filepaths = fs.ls(
            f"{data['author']}/{data['repo']}/{sub}", revision=data["ref"], detail=False
        )
        filepath = choose_diffusers_weights(filepaths, prefer_fp16=prefer_fp16)
        if not filepath:
            msg = f"Could not find any weights in {base_url}/{sub}"
            raise ValueError(msg)
        filename = filepath.split("/")[-1]
    url = f"{base_url}{sub}/{filename}".replace("/tree/", "/resolve/")
    new_path = get_cached_url_path(url, category="weights")
    return new_path


def choose_diffusers_weights(filenames, prefer_fp16=True):
    extension_priority = (".safetensors", ".bin", ".pth", ".pt")
    # filter out any files that don't have a valid extension
    filenames = [f for f in filenames if any(f.endswith(e) for e in extension_priority)]
    filenames_and_extension = [(f, os.path.splitext(f)[1]) for f in filenames]
    # sort by priority
    if prefer_fp16:
        filenames_and_extension.sort(
            key=lambda x: ("fp16" not in x[0], extension_priority.index(x[1]))
        )
    else:
        filenames_and_extension.sort(
            key=lambda x: ("fp16" in x[0], extension_priority.index(x[1]))
        )
    if filenames_and_extension:
        return filenames_and_extension[0][0]
    return None


def load_sd15_diffusers_weights(base_url: str, device=None):
    from imaginairy.utils import get_device
    from imaginairy.weight_management.conversion import cast_weights
    from imaginairy.weight_management.utils import (
        COMPONENT_NAMES,
        FORMAT_NAMES,
        MODEL_NAMES,
    )

    base_url = normalize_diffusers_repo_url(base_url)
    if device is None:
        device = get_device()
    vae_weights_path = download_diffusers_weights(base_url=base_url, sub="vae")
    vae_weights = open_weights(vae_weights_path, device=device)
    vae_weights = cast_weights(
        source_weights=vae_weights,
        source_model_name=MODEL_NAMES.SD15,
        source_component_name=COMPONENT_NAMES.VAE,
        source_format=FORMAT_NAMES.DIFFUSERS,
        dest_format=FORMAT_NAMES.REFINERS,
    )

    unet_weights_path = download_diffusers_weights(base_url=base_url, sub="unet")
    unet_weights = open_weights(unet_weights_path, device=device)
    unet_weights = cast_weights(
        source_weights=unet_weights,
        source_model_name=MODEL_NAMES.SD15,
        source_component_name=COMPONENT_NAMES.UNET,
        source_format=FORMAT_NAMES.DIFFUSERS,
        dest_format=FORMAT_NAMES.REFINERS,
    )

    text_encoder_weights_path = download_diffusers_weights(
        base_url=base_url, sub="text_encoder"
    )
    text_encoder_weights = open_weights(text_encoder_weights_path, device=device)
    text_encoder_weights = cast_weights(
        source_weights=text_encoder_weights,
        source_model_name=MODEL_NAMES.SD15,
        source_component_name=COMPONENT_NAMES.TEXT_ENCODER,
        source_format=FORMAT_NAMES.DIFFUSERS,
        dest_format=FORMAT_NAMES.REFINERS,
    )

    return vae_weights, unet_weights, text_encoder_weights


def load_sdxl_diffusers_weights(base_url: str, device=None, dtype=torch.float16):
    from imaginairy.utils import get_device

    device = device or get_device()

    base_url = normalize_diffusers_repo_url(base_url)

    translator = translators.diffusers_autoencoder_kl_to_refiners_translator()
    vae_weights_path = download_diffusers_weights(
        base_url=base_url, sub="vae", prefer_fp16=False
    )
    logger.debug(f"vae: {vae_weights_path}")
    vae_weights = translator.load_and_translate_weights(
        source_path=vae_weights_path,
        device="cpu",
    )
    lda = SDXLAutoencoderSliced(device="cpu", dtype=dtype)
    lda.load_state_dict(vae_weights)
    del vae_weights

    translator = translators.diffusers_unet_sdxl_to_refiners_translator()
    unet_weights_path = download_diffusers_weights(
        base_url=base_url, sub="unet", prefer_fp16=True
    )
    logger.debug(f"unet: {unet_weights_path}")
    unet_weights = translator.load_and_translate_weights(
        source_path=unet_weights_path,
        device="cpu",
    )
    unet = SDXLUNet(device="cpu", dtype=dtype, in_channels=4)
    unet.load_state_dict(unet_weights)
    del unet_weights

    text_encoder_1_path = download_diffusers_weights(
        base_url=base_url, sub="text_encoder"
    )
    text_encoder_2_path = download_diffusers_weights(
        base_url=base_url, sub="text_encoder_2"
    )
    logger.debug(f"text encoder 1: {text_encoder_1_path}")
    logger.debug(f"text encoder 2: {text_encoder_2_path}")
    text_encoder_weights = (
        translators.DoubleTextEncoderTranslator().load_and_translate_weights(
            text_encoder_l_weights_path=text_encoder_1_path,
            text_encoder_g_weights_path=text_encoder_2_path,
            device="cpu",
        )
    )
    text_encoder = DoubleTextEncoder(device="cpu", dtype=dtype)
    text_encoder.load_state_dict(text_encoder_weights)
    del text_encoder_weights
    lda = lda.to(device=device)
    unet = unet.to(device=device)
    text_encoder = text_encoder.to(device=device)
    sd = StableDiffusion_XL(
        device=device, dtype=dtype, lda=lda, unet=unet, clip_text_encoder=text_encoder
    )
    sd.lda.to(device=device, dtype=torch.float32)

    return sd


def load_sdxl_pipeline(base_url, device=None):
    logger.info(f"Loading SDXL weights from {base_url}")
    device = device or get_device()
    sd = load_sdxl_diffusers_weights(base_url, device=device)

    sd.set_self_attention_guidance(enable=True)

    return sd


def open_weights(filepath, device=None):
    from imaginairy.utils import get_device

    if device is None:
        device = get_device()

    if "safetensor" in filepath.lower():
        from refiners.fluxion.utils import safe_open

        with safe_open(path=filepath, framework="pytorch", device=device) as tensors:
            state_dict = {
                key: tensors.get_tensor(key)
                for key in tensors.keys()  # noqa
            }
    else:
        import torch

        state_dict = torch.load(filepath, map_location=device)

    while "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    return state_dict


def load_tensors(tensorfile, map_location=None):
    if tensorfile == "empty":
        # used for testing
        return {}
    if tensorfile.endswith((".ckpt", ".pth", ".bin")):
        return torch.load(tensorfile, map_location=map_location)
    if tensorfile.endswith(".safetensors"):
        return load_file(tensorfile, device=map_location)

    return load_file(tensorfile, device=map_location)

    # raise ValueError(f"Unknown tensorfile type: {tensorfile}")


def load_stable_diffusion_compvis_weights(weights_url):
    from imaginairy.utils import get_device
    from imaginairy.weight_management.conversion import cast_weights
    from imaginairy.weight_management.utils import (
        COMPONENT_NAMES,
        FORMAT_NAMES,
        MODEL_NAMES,
    )

    weights_path = get_cached_url_path(weights_url, category="weights")
    logger.info(f"Loading weights from {weights_path}")
    state_dict = open_weights(weights_path, device=get_device())

    text_encoder_prefix = "cond_stage_model."
    cut_start = len(text_encoder_prefix)
    text_encoder_state_dict = {
        k[cut_start:]: v
        for k, v in state_dict.items()
        if k.startswith(text_encoder_prefix)
    }
    text_encoder_state_dict = cast_weights(
        source_weights=text_encoder_state_dict,
        source_model_name=MODEL_NAMES.SD15,
        source_component_name=COMPONENT_NAMES.TEXT_ENCODER,
        source_format=FORMAT_NAMES.COMPVIS,
        dest_format=FORMAT_NAMES.DIFFUSERS,
    )
    text_encoder_state_dict = cast_weights(
        source_weights=text_encoder_state_dict,
        source_model_name=MODEL_NAMES.SD15,
        source_component_name=COMPONENT_NAMES.TEXT_ENCODER,
        source_format=FORMAT_NAMES.DIFFUSERS,
        dest_format=FORMAT_NAMES.REFINERS,
    )

    vae_prefix = "first_stage_model."
    cut_start = len(vae_prefix)
    vae_state_dict = {
        k[cut_start:]: v for k, v in state_dict.items() if k.startswith(vae_prefix)
    }
    vae_state_dict = cast_weights(
        source_weights=vae_state_dict,
        source_model_name=MODEL_NAMES.SD15,
        source_component_name=COMPONENT_NAMES.VAE,
        source_format=FORMAT_NAMES.COMPVIS,
        dest_format=FORMAT_NAMES.DIFFUSERS,
    )
    vae_state_dict = cast_weights(
        source_weights=vae_state_dict,
        source_model_name=MODEL_NAMES.SD15,
        source_component_name=COMPONENT_NAMES.VAE,
        source_format=FORMAT_NAMES.DIFFUSERS,
        dest_format=FORMAT_NAMES.REFINERS,
    )

    unet_prefix = "model."
    cut_start = len(unet_prefix)
    unet_state_dict = {
        k[cut_start:]: v for k, v in state_dict.items() if k.startswith(unet_prefix)
    }
    unet_state_dict = cast_weights(
        source_weights=unet_state_dict,
        source_model_name=MODEL_NAMES.SD15,
        source_component_name=COMPONENT_NAMES.UNET,
        source_format=FORMAT_NAMES.COMPVIS,
        dest_format=FORMAT_NAMES.DIFFUSERS,
    )

    unet_state_dict = cast_weights(
        source_weights=unet_state_dict,
        source_model_name=MODEL_NAMES.SD15,
        source_component_name=COMPONENT_NAMES.UNET,
        source_format=FORMAT_NAMES.DIFFUSERS,
        dest_format=FORMAT_NAMES.REFINERS,
    )

    return vae_state_dict, unet_state_dict, text_encoder_state_dict
