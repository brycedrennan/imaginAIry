from functools import lru_cache

import numpy as np
import torch
from diffusers.pipelines.stable_diffusion import safety_checker as safety_checker_mod
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from transformers import AutoFeatureExtractor


@lru_cache()
def safety_models():
    safety_model_id = "CompVis/stable-diffusion-safety-checker"
    monkeypatch_safety_cosine_distance()
    safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)
    return safety_feature_extractor, safety_checker


@lru_cache()
def monkeypatch_safety_cosine_distance():
    orig_cosine_distance = safety_checker_mod.cosine_distance

    def cosine_distance_float32(image_embeds, text_embeds):
        """
        In some environments we need to distance to be in float32
        but it was coming as BFloat16
        """
        return orig_cosine_distance(image_embeds, text_embeds).to(torch.float32)

    safety_checker_mod.cosine_distance = cosine_distance_float32


def is_nsfw(img):
    safety_feature_extractor, safety_checker = safety_models()
    safety_checker_input = safety_feature_extractor([img], return_tensors="pt")
    clip_input = safety_checker_input.pixel_values

    _, has_nsfw_concept = safety_checker(
        images=[np.empty((2, 2))], clip_input=clip_input
    )

    return has_nsfw_concept[0]
