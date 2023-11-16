import safetensors

from imaginairy.model_manager import (
    get_cached_url_path,
    open_weights,
    resolve_model_paths,
)
from imaginairy.weight_management import utils
from imaginairy.weight_management.pattern_collapse import find_state_dict_key_patterns
from imaginairy.weight_management.utils import save_model_info


def save_compvis_patterns():
    (
        model_metadata,
        weights_url,
        config_path,
        control_weights_paths,
    ) = resolve_model_paths(
        weights_path="openjourney-v1",
    )
    weights_path = get_cached_url_path(weights_url, category="weights")

    with safetensors.safe_open(weights_path, "pytorch") as f:
        weights_keys = f.keys()

    text_encoder_prefix = "cond_stage_model.transformer.text_model"
    text_encoder_keys = [k for k in weights_keys if k.startswith(text_encoder_prefix)]
    save_weight_info(
        model_name=utils.MODEL_NAMES.SD15,
        component_name=utils.COMPONENT_NAMES.TEXT_ENCODER,
        format_name=utils.FORMAT_NAMES.COMPVIS,
        weights_keys=text_encoder_keys,
    )

    vae_prefix = "first_stage_model"
    vae_keys = [k for k in weights_keys if k.startswith(vae_prefix)]
    save_weight_info(
        model_name=utils.MODEL_NAMES.SD15,
        component_name=utils.COMPONENT_NAMES.VAE,
        format_name=utils.FORMAT_NAMES.COMPVIS,
        weights_keys=vae_keys,
    )

    unet_prefix = "model.diffusion_model"
    unet_keys = [k for k in weights_keys if k.startswith(unet_prefix)]
    save_weight_info(
        model_name=utils.MODEL_NAMES.SD15,
        component_name=utils.COMPONENT_NAMES.UNET,
        format_name=utils.FORMAT_NAMES.COMPVIS,
        weights_keys=unet_keys,
    )


def save_diffusers_patterns():
    save_weight_info(
        model_name=utils.MODEL_NAMES.SD15,
        component_name=utils.COMPONENT_NAMES.VAE,
        format_name=utils.FORMAT_NAMES.DIFFUSERS,
        weights_url="https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors",
    )

    save_weight_info(
        model_name=utils.MODEL_NAMES.SD15,
        component_name=utils.COMPONENT_NAMES.UNET,
        format_name=utils.FORMAT_NAMES.DIFFUSERS,
        weights_url="https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors",
    )

    save_weight_info(
        model_name=utils.MODEL_NAMES.SD15,
        component_name=utils.COMPONENT_NAMES.TEXT_ENCODER,
        format_name=utils.FORMAT_NAMES.DIFFUSERS,
        weights_url="https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/text_encoder/model.fp16.safetensors",
    )


def save_lora_patterns():
    filepath = "/Users/bryce/projects/sandbox-img-gen/refiners/weights/pytorch_lora_weights-refiners.safetensors"
    state_dict = open_weights(filepath, device="cpu")

    save_weight_info(
        model_name=utils.MODEL_NAMES.SD15,
        component_name=utils.COMPONENT_NAMES.LORA,
        format_name=utils.FORMAT_NAMES.REFINERS,
        weights_keys=list(state_dict.keys()),
    )

    save_weight_info(
        model_name=utils.MODEL_NAMES.SD15,
        component_name=utils.COMPONENT_NAMES.LORA,
        format_name=utils.FORMAT_NAMES.DIFFUSERS,
        weights_url="https://huggingface.co/pcuenq/pokemon-lora/resolve/main/pytorch_lora_weights.bin",
    )


def save_weight_info(
    model_name, component_name, format_name, weights_url=None, weights_keys=None
):
    if weights_keys is None and weights_url is None:
        msg = "Either weights_keys or weights_url must be provided"
        raise ValueError(msg)

    if weights_keys is None:
        weights_path = get_cached_url_path(weights_url, category="weights")

        state_dict = open_weights(weights_path, device="cpu")
        weights_keys = list(state_dict.keys())

    # prefixes = utils.prefixes_only(weights_keys)

    save_model_info(
        model_name=model_name,
        component_name=component_name,
        format_name=format_name,
        info_type="weights_keys",
        data=weights_keys,
    )

    patterns = find_state_dict_key_patterns(weights_keys)
    save_model_info(
        model_name=model_name,
        component_name=component_name,
        format_name=format_name,
        info_type="patterns",
        data=patterns,
    )


def save_patterns():
    save_lora_patterns()
    # save_compvis_patterns()
    # save_diffusers_patterns()


if __name__ == "__main__":
    save_patterns()
