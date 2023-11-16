import os.path

_base_dir = os.path.dirname(os.path.realpath(__file__))

WEIGHT_MAPS_PATH = os.path.join(_base_dir, "maps")
WEIGHT_INFO_PATH = os.path.join(_base_dir, "weight_info")


class MODEL_NAMES:
    SD15 = "stable-diffusion-1-5"


class COMPONENT_NAMES:
    VAE = "vae"
    TEXT_ENCODER = "text"
    UNET = "unet"
    LORA = "lora"


class FORMAT_NAMES:
    COMPVIS = "compvis"
    DIFFUSERS = "diffusers"
    REFINERS = "refiners"


def save_model_info(model_name, component_name, format_name, info_type, data):
    import json

    model_name = model_name.replace("_", "-")
    component_name = component_name.replace("_", "-")
    format_name = format_name.replace("_", "-")
    filename = os.path.join(
        WEIGHT_INFO_PATH,
        f"{model_name}_{component_name}_{format_name}.{info_type}.json",
    )
    with open(filename, "w") as f:
        f.write(json.dumps(data, indent=2))


def prefixes_only(keys):
    new_keys = []
    prev_key = None
    for k in keys:
        new_key = k.rsplit(".", 1)[0]
        if new_key != prev_key:
            new_keys.append(new_key)
        prev_key = new_key
    return new_keys
