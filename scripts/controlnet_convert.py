import os

import torch
from safetensors.torch import load_file, save_file

from imaginairy.utils.downloads import get_cached_url_path
from imaginairy.utils.paths import PKG_ROOT

sd15_url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/889b629140e71758e1e0006e355c331a5744b4bf/v1-5-pruned-emaonly.ckpt"


def main():
    """Script to convert the controlnet weights into diffs that are ready to be applied to any s1.5 weights."""

    control_types = [
        "canny",
        "depth",
        "hed",
        "mlsd",
        "normal",
        "openpose",
        "scribble",
        "seg",
    ]
    url_template = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_{control_type}.pth"
    urls = {
        control_type: url_template.format(control_type=control_type)
        for control_type in control_types
    }
    dest = f"{PKG_ROOT}/../other/weights/controlnet"

    for control_type, url in urls.items():
        print(f"Downloading {control_type} weights from {url}")

        out_filepath = extract_controlnet_essence(
            control_type=control_type,
            controlnet_url=url,
            dest_folder=dest,
        )

        sd15_path = get_cached_url_path(sd15_url)
        sd15_state_dict = torch.load(sd15_path, map_location="cpu")
        sd15_state_dict = sd15_state_dict.get("state_dict", sd15_state_dict)
        reconstituted_controlnet_statedict = apply_controlnet(
            base_state_dict=sd15_state_dict,
            controlnet_state_dict=load_file(out_filepath),
        )

        controlnet_path = get_cached_url_path(url)
        import time

        time.sleep(1)
        controlnet_statedict = torch.load(controlnet_path, map_location="cpu")
        print("\n\nComparing reconstructed controlnet with original")
        for k in controlnet_statedict:
            if k not in reconstituted_controlnet_statedict:
                print(f"Key {k} not in reconstituted")
            elif (
                controlnet_statedict[k].shape
                != reconstituted_controlnet_statedict[k].shape
            ):
                print(f"Key {k} has different shape")
                print(controlnet_statedict[k].shape)
                print(reconstituted_controlnet_statedict[k].shape)
            else:
                diff = controlnet_statedict[k] - reconstituted_controlnet_statedict[k]
                diff_sum = torch.abs(diff).sum()
                if diff_sum > 3.467949682089966e-05:
                    print(f"Key {k} has different values {diff_sum}")


def extract_controlnet_essence(control_type, controlnet_url, dest_folder):
    print(f"Extracting essence of {control_type} weights from {controlnet_url}")
    outpath = f"{dest_folder}/controlnet15_diff_{control_type}.safetensors"
    if os.path.exists(outpath):
        print(f"File {outpath} already exists, skipping")
        return outpath
    os.makedirs(dest_folder, exist_ok=True)
    sd15_path = get_cached_url_path(sd15_url)
    controlnet_path = get_cached_url_path(controlnet_url)
    print(f"sd15_path: {sd15_path}")
    print(f"controlnet_path: {controlnet_path}")

    sd15_state_dict = torch.load(sd15_path, map_location="cpu")
    sd15_state_dict = sd15_state_dict.get("state_dict", sd15_state_dict)

    controlnet_state_dict = torch.load(controlnet_path, map_location="cpu")
    controlnet_state_dict = controlnet_state_dict.get(
        "state_dict", controlnet_state_dict
    )

    final_state_dict = {}
    skip_prefixes = ("first_stage_model", "cond_stage_model")
    for key in controlnet_state_dict:
        if key.startswith(skip_prefixes):
            continue

        if key.startswith("control_"):
            sd15_key_name = "model.diffusion_" + key[len("control_") :]
        else:
            sd15_key_name = key

        if sd15_key_name in sd15_state_dict:
            diff_value = controlnet_state_dict[key] - sd15_state_dict[sd15_key_name]
            final_state_dict[key] = diff_value
        else:
            final_state_dict[key] = controlnet_state_dict[key]
    save_file(final_state_dict, outpath)
    return outpath


def apply_controlnet(base_state_dict, controlnet_state_dict):
    for key in controlnet_state_dict:
        if key.startswith("control_"):
            sd15_key_name = "model.diffusion_" + key[len("control_") :]
        else:
            sd15_key_name = key

        if sd15_key_name in base_state_dict:
            b = base_state_dict[sd15_key_name]
            c_diff = controlnet_state_dict[key]
            new_c = b + c_diff
            base_state_dict[key] = new_c
        else:
            base_state_dict[key] = controlnet_state_dict[key]
    return base_state_dict


if __name__ == "__main__":
    main()
