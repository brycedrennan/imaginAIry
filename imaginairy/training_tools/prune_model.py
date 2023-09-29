import logging
import os

import torch

logger = logging.getLogger(__name__)


def prune_diffusion_ckpt(ckpt_path, dst_path=None):
    if dst_path is None:
        dst_path = f"{os.path.splitext(ckpt_path)[0]}-pruned.ckpt"

    data = torch.load(ckpt_path, map_location="cpu")

    new_data = prune_model_data(data)

    torch.save(new_data, dst_path)

    size_initial = os.path.getsize(ckpt_path)
    newsize = os.path.getsize(dst_path)
    msg = (
        f"New ckpt size: {newsize * 1e-9:.2f} GB. "
        f"Saved {(size_initial - newsize) * 1e-9:.2f} GB by removing optimizer states"
    )
    logger.info(msg)


def prune_model_data(data, only_keep_ema=True):
    data.pop("optimizer_states", None)
    if only_keep_ema:
        state_dict = data["state_dict"]
        model_keys = [k for k in state_dict if k.startswith("model.")]

        for model_key in model_keys:
            ema_key = "model_ema." + model_key[6:].replace(".", "")
            state_dict[model_key] = state_dict[ema_key]
            del state_dict[ema_key]

    return data
