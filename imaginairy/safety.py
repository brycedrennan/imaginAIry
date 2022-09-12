from functools import lru_cache

from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from transformers import AutoFeatureExtractor


@lru_cache()
def safety_models():
    safety_model_id = "CompVis/stable-diffusion-safety-checker"
    safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)
    return safety_feature_extractor, safety_checker


def is_nsfw(img, x_sample, half_mode=False):
    safety_feature_extractor, safety_checker = safety_models()
    safety_checker_input = safety_feature_extractor([img], return_tensors="pt")
    clip_input = safety_checker_input.pixel_values

    _, has_nsfw_concept = safety_checker(
        images=x_sample[None, :], clip_input=clip_input
    )
    return has_nsfw_concept[0]
