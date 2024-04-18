"""Classes and functions for managing AI models"""

import logging
import os
import sys
from functools import lru_cache

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file

from imaginairy import config as iconfig
from imaginairy.config import IMAGE_WEIGHTS_SHORT_NAMES, ModelArchitecture
from imaginairy.modules import attention
from imaginairy.modules.refiners_sd import (
    SDXLAutoencoderSliced,
    StableDiffusion_XL,
    StableDiffusion_XL_Inpainting,
)
from imaginairy.utils import clear_gpu_cache, get_device, instantiate_from_config
from imaginairy.utils.downloads import (
    HuggingFaceAuthorizationError,
    download_huggingface_weights,
    get_cached_url_path,
    is_diffusers_repo_url,
    normalize_diffusers_repo_url,
    resolve_path_or_url,
)
from imaginairy.utils.model_cache import memory_managed_model
from imaginairy.utils.named_resolutions import normalize_image_size
from imaginairy.utils.paths import PKG_ROOT
from imaginairy.vendored.refiners.foundationals.clip.text_encoder import (
    CLIPTextEncoderL,
)
from imaginairy.vendored.refiners.foundationals.latent_diffusion import (
    DoubleTextEncoder,
    SD1UNet,
    SDXLUNet,
)
from imaginairy.vendored.refiners.foundationals.latent_diffusion.model import (
    LatentDiffusionModel,
)
from imaginairy.weight_management import translators
from imaginairy.weight_management.translators import (
    DoubleTextEncoderTranslator,
    diffusers_autoencoder_kl_to_refiners_translator,
    diffusers_unet_sdxl_to_refiners_translator,
    load_weight_map,
)

logger = logging.getLogger(__name__)

MOST_RECENTLY_LOADED_MODEL = None


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

    sd = _get_diffusion_model_refiners(
        weights_location=weights_config.weights_location,
        architecture_alias=weights_config.architecture.primary_alias,
        for_inpainting=for_inpainting,
        dtype=dtype,
    )
    # ensures a "fresh" copy that doesn't have additional injected parts
    sd = sd.structural_copy()

    sd.set_self_attention_guidance(enable=True)

    return sd


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
        sd = load_sd15_pipeline(
            weights_location=weights_location,
            for_inpainting=for_inpainting,
            device=device,
            dtype=dtype,
        )
    elif architecture.primary_alias in ("sdxl", "sdxlinpaint"):
        sd = load_sdxl_pipeline(
            base_url=weights_location, device=device, for_inpainting=for_inpainting
        )
    else:
        msg = f"Invalid architecture {architecture.primary_alias}"
        raise ValueError(msg)

    MOST_RECENTLY_LOADED_MODEL = sd

    msg = (
        "Pipeline loaded "
        f"sd[dtype:{sd.dtype} device:{sd.device}] "
        f"sd.unet[dtype:{sd.unet.dtype} device:{sd.unet.device}] "
        f"sd.lda[dtype:{sd.lda.dtype} device:{sd.lda.device}]"
        f"sd.clip_text_encoder[dtype:{sd.clip_text_encoder.dtype} device:{sd.clip_text_encoder.device}]"
    )
    logger.debug(msg)

    return sd


# new
def load_sd15_pipeline(
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
        ) = load_sd15_diffusers_weights(weights_location, device="cpu")
    else:
        (
            vae_weights,
            unet_weights,
            text_encoder_weights,
        ) = load_stable_diffusion_compvis_weights(weights_location)
    StableDiffusionCls: type[LatentDiffusionModel]
    if for_inpainting:
        unet = SD1UNet(in_channels=9, device="cpu", dtype=dtype)
        StableDiffusionCls = StableDiffusion_1_Inpainting
    else:
        unet = SD1UNet(in_channels=4, device="cpu", dtype=dtype)
        StableDiffusionCls = StableDiffusion_1
    logger.debug(f"Using class {StableDiffusionCls.__name__}")

    sd = StableDiffusionCls(
        device=device, dtype=dtype, lda=SD1AutoencoderSliced(), unet=unet
    )
    logger.debug("Loading VAE")
    sd.lda.load_state_dict(vae_weights, assign=True)

    logger.debug("Loading text encoder")
    sd.clip_text_encoder.load_state_dict(text_encoder_weights, assign=True)

    logger.debug("Loading UNet")
    sd.unet.load_state_dict(unet_weights, strict=False, assign=True)

    logger.debug(f"'{weights_location}' Loaded")
    sd.to(device=device, dtype=dtype)
    return sd


def _get_sd15_diffusion_model_refiners_new(
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
        ) = load_sd15_diffusers_weights(weights_location, device="cpu")
    else:
        (
            vae_weights,
            unet_weights,
            text_encoder_weights,
        ) = load_stable_diffusion_compvis_weights(weights_location)
    StableDiffusionCls: type[LatentDiffusionModel]
    if for_inpainting:
        unet = SD1UNet(in_channels=9, device="cpu", dtype=dtype)
        StableDiffusionCls = StableDiffusion_1_Inpainting
    else:
        unet = SD1UNet(in_channels=4, device="cpu", dtype=dtype)
        StableDiffusionCls = StableDiffusion_1

    logger.debug("Loading UNet")
    unet.load_state_dict(unet_weights, strict=False, assign=True)
    del unet_weights
    unet.to(device=device, dtype=dtype)

    logger.debug("Loading VAE")
    lda = SD1AutoencoderSliced(device=device, dtype=dtype)
    lda.load_state_dict(vae_weights, assign=True)
    del vae_weights
    lda.to(device=device, dtype=dtype)

    logger.debug("Loading text encoder")
    clip_text_encoder = CLIPTextEncoderL()
    clip_text_encoder.load_state_dict(text_encoder_weights, assign=True)
    del text_encoder_weights
    clip_text_encoder.to(device=device, dtype=dtype)

    logger.debug(f"Using class {StableDiffusionCls.__name__}")

    sd = StableDiffusionCls(device=None, dtype=dtype, lda=lda, unet=unet)  # type: ignore
    sd.to(device=device, dtype=dtype)

    logger.debug(f"'{weights_location}' Loaded")
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
    controlnet.load_state_dict(controlnet_state_dict, assign=True)
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
    vae_weights_path = download_huggingface_weights(base_url=base_url, sub="vae")
    vae_weights = open_weights(vae_weights_path, device=device)
    vae_weights = cast_weights(
        source_weights=vae_weights,
        source_model_name=MODEL_NAMES.SD15,
        source_component_name=COMPONENT_NAMES.VAE,
        source_format=FORMAT_NAMES.DIFFUSERS,
        dest_format=FORMAT_NAMES.REFINERS,
    )

    unet_weights_path = download_huggingface_weights(base_url=base_url, sub="unet")
    unet_weights = open_weights(unet_weights_path, device=device)
    unet_weights = cast_weights(
        source_weights=unet_weights,
        source_model_name=MODEL_NAMES.SD15,
        source_component_name=COMPONENT_NAMES.UNET,
        source_format=FORMAT_NAMES.DIFFUSERS,
        dest_format=FORMAT_NAMES.REFINERS,
    )

    text_encoder_weights_path = download_huggingface_weights(
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
    first_vae = next(iter(vae_weights.values()))
    first_unet = next(iter(unet_weights.values()))
    first_encoder = next(iter(text_encoder_weights.values()))
    msg = (
        f"vae weights. dtype: {first_vae.dtype} device: {first_vae.device}\n"
        f"unet weights. dtype: {first_unet.dtype} device: {first_unet.device}\n"
        f"text_encoder weights. dtype: {first_encoder.dtype} device: {first_encoder.device}\n"
    )
    logger.debug(msg)
    return vae_weights, unet_weights, text_encoder_weights


def load_sdxl_pipeline_from_diffusers_weights(
    base_url: str, for_inpainting=False, device=None, dtype=torch.float16
):
    from imaginairy.utils import get_device

    device = device or get_device()

    base_url = normalize_diffusers_repo_url(base_url)

    translator = translators.diffusers_autoencoder_kl_to_refiners_translator()
    vae_weights_path = download_huggingface_weights(
        base_url=base_url, sub="vae", prefer_fp16=False
    )
    logger.debug(f"vae: {vae_weights_path}")
    vae_weights = translator.load_and_translate_weights(
        source_path=vae_weights_path,
        device="cpu",
    )
    lda = SDXLAutoencoderSliced(device="cpu", dtype=dtype)
    lda.load_state_dict(vae_weights, assign=True)
    del vae_weights

    translator = translators.diffusers_unet_sdxl_to_refiners_translator()
    unet_weights_path = download_huggingface_weights(
        base_url=base_url, sub="unet", prefer_fp16=True
    )
    logger.debug(f"unet: {unet_weights_path}")
    unet_weights = translator.load_and_translate_weights(
        source_path=unet_weights_path,
        device="cpu",
    )
    if for_inpainting:
        unet = SDXLUNet(device="cpu", dtype=dtype, in_channels=9)
    else:
        unet = SDXLUNet(device="cpu", dtype=dtype, in_channels=4)
    unet.load_state_dict(unet_weights, assign=True)
    del unet_weights

    text_encoder_1_path = download_huggingface_weights(
        base_url=base_url, sub="text_encoder"
    )
    text_encoder_2_path = download_huggingface_weights(
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
    text_encoder = DoubleTextEncoder(device="cpu", dtype=torch.float32)
    text_encoder.load_state_dict(text_encoder_weights, assign=True)
    del text_encoder_weights
    lda = lda.to(device=device, dtype=torch.float32)
    unet = unet.to(device=device, dtype=dtype)
    text_encoder = text_encoder.to(device=device, dtype=dtype)
    if for_inpainting:
        StableDiffusionCls = StableDiffusion_XL_Inpainting
    else:
        StableDiffusionCls = StableDiffusion_XL

    sd = StableDiffusionCls(
        device=device, dtype=None, lda=lda, unet=unet, clip_text_encoder=text_encoder
    )

    return sd


def load_sdxl_pipeline_from_compvis_weights(
    base_url: str, for_inpainting=False, device=None, dtype=torch.float16
):
    from imaginairy.utils import get_device

    device = device or get_device()
    unet_weights, vae_weights, text_encoder_weights = load_sdxl_compvis_weights(
        base_url
    )
    lda = SDXLAutoencoderSliced(device="cpu", dtype=dtype)
    lda.load_state_dict(vae_weights, assign=True)
    del vae_weights

    if for_inpainting:
        unet = SDXLUNet(device="cpu", dtype=dtype, in_channels=9)
    else:
        unet = SDXLUNet(device="cpu", dtype=dtype, in_channels=4)
    unet.load_state_dict(unet_weights, assign=True)
    del unet_weights

    text_encoder = DoubleTextEncoder(device="cpu", dtype=torch.float32)
    text_encoder.load_state_dict(text_encoder_weights, assign=True)
    del text_encoder_weights
    lda = lda.to(device=device, dtype=torch.float32)
    unet = unet.to(device=device)
    text_encoder = text_encoder.to(device=device)

    if for_inpainting:
        StableDiffusionCls = StableDiffusion_XL_Inpainting
    else:
        StableDiffusionCls = StableDiffusion_XL
    sd = StableDiffusionCls(
        device=device, dtype=None, lda=lda, unet=unet, clip_text_encoder=text_encoder
    )

    return sd


def load_sdxl_pipeline(base_url, device=None, for_inpainting=False):
    logger.info(f"Loading SDXL weights from {base_url}")
    device = device or get_device()

    with logger.timed_info(f"Loaded SDXL pipeline from {base_url}"):
        if is_diffusers_repo_url(base_url):
            sd = load_sdxl_pipeline_from_diffusers_weights(
                base_url, for_inpainting=for_inpainting, device=device
            )
        else:
            sd = load_sdxl_pipeline_from_compvis_weights(
                base_url, for_inpainting=for_inpainting, device=device
            )
        return sd


def open_weights(filepath, device=None):
    from imaginairy.utils import get_device

    if device is None:
        device = get_device()

    if "safetensor" in filepath.lower():
        from imaginairy.vendored.refiners.fluxion.utils import safe_open

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


def load_sdxl_compvis_weights(url):
    from safetensors import safe_open

    weights_path = resolve_path_or_url(url)
    state_dict = {}
    unet_state_dict = {}
    vae_state_dict = {}
    text_encoder_1_state_dict = {}
    text_encoder_2_state_dict = {}
    with safe_open(weights_path, framework="pt") as f:
        for key in f.keys():  # noqa
            if key.startswith("model.diffusion_model."):
                unet_state_dict[key] = f.get_tensor(key)
            elif key.startswith("first_stage_model"):
                vae_state_dict[key] = f.get_tensor(key)
            elif key.startswith("conditioner.embedders.0."):
                text_encoder_1_state_dict[key] = f.get_tensor(key)
            elif key.startswith("conditioner.embedders.1."):
                text_encoder_2_state_dict[key] = f.get_tensor(key)
            else:
                state_dict[key] = f.get_tensor(key)
                logger.warning(f"Unused key {key}")

    unet_weightmap = load_weight_map("Compvis-UNet-SDXL-to-Diffusers")
    vae_weightmap = load_weight_map("Compvis-Autoencoder-SDXL-to-Diffusers")
    text_encoder_1_weightmap = load_weight_map("Compvis-TextEncoder-SDXL-to-Diffusers")
    text_encoder_2_weightmap = load_weight_map(
        "Compvis-OpenClipTextEncoder-SDXL-to-Diffusers"
    )

    diffusers_unet_state_dict = unet_weightmap.translate_weights(unet_state_dict)

    refiners_unet_state_dict = (
        diffusers_unet_sdxl_to_refiners_translator().translate_weights(
            diffusers_unet_state_dict
        )
    )

    diffusers_vae_state_dict = vae_weightmap.translate_weights(vae_state_dict)

    refiners_vae_state_dict = (
        diffusers_autoencoder_kl_to_refiners_translator().translate_weights(
            diffusers_vae_state_dict
        )
    )

    diffusers_text_encoder_1_state_dict = text_encoder_1_weightmap.translate_weights(
        text_encoder_1_state_dict
    )

    for key in list(text_encoder_2_state_dict.keys()):
        if key.endswith((".in_proj_bias", ".in_proj_weight")):
            value = text_encoder_2_state_dict[key]
            q, k, v = value.chunk(3, dim=0)
            text_encoder_2_state_dict[f"{key}.0"] = q
            text_encoder_2_state_dict[f"{key}.1"] = k
            text_encoder_2_state_dict[f"{key}.2"] = v
            del text_encoder_2_state_dict[key]

    diffusers_text_encoder_2_state_dict = text_encoder_2_weightmap.translate_weights(
        text_encoder_2_state_dict
    )

    refiners_text_encoder_weights = DoubleTextEncoderTranslator().translate_weights(
        diffusers_text_encoder_1_state_dict, diffusers_text_encoder_2_state_dict
    )
    return (
        refiners_unet_state_dict,
        refiners_vae_state_dict,
        refiners_text_encoder_weights,
    )
