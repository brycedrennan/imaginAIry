import logging
from functools import lru_cache

import torch
from diffusers.pipelines.stable_diffusion import safety_checker as safety_checker_mod
from transformers import AutoFeatureExtractor

from imaginairy.enhancers.blur_detect import is_blurry
from imaginairy.schema import SafetyMode

logger = logging.getLogger(__name__)


class SafetyResult:
    # increase this value to create a stronger `nfsw` filter
    # at the cost of increasing the possibility of filtering benign images
    _default_adjustment = 0.0

    def __init__(self):
        self.nsfw_scores = {}
        self.special_care_scores = {}
        self.is_filtered = False

    def add_special_care_score(self, concept_idx, abs_score, threshold):
        adjustment = self._default_adjustment
        adjusted_score = round(abs_score - threshold + adjustment, 3)
        try:
            score_name = _SPECIAL_CARE_DESCRIPTIONS[concept_idx]
        except LookupError:
            score_name = ""
        if adjusted_score > 0 and score_name:
            logger.debug(
                f"    🔞🔞 '{score_name}' abs:{abs_score:.3f} adj:{adjusted_score}"
            )
        self.special_care_scores[concept_idx] = adjusted_score

    def add_nsfw_score(self, concept_idx, abs_score, threshold):
        if len(self.special_care_scores) != 3:
            raise ValueError("special care scores must be set first")
        adjustment = self._default_adjustment
        if self.special_care_score > 0:
            adjustment += 0.01
        adjusted_score = round(abs_score - threshold + adjustment, 3)
        try:
            score_name = _CONCEPT_DESCRIPTIONS[concept_idx]
        except LookupError:
            score_name = ""
        if adjusted_score > 0 and score_name:
            logger.debug(
                f"    🔞 '{concept_idx}:{score_name}' abs:{abs_score:.3f} adj:{adjusted_score}"
            )
        self.nsfw_scores[concept_idx] = adjusted_score

    @property
    def nsfw_score(self):
        return max(self.nsfw_scores.values())

    @property
    def special_care_score(self):
        return max(self.special_care_scores.values())

    @property
    def special_care_nsfw_score(self):
        return min(self.nsfw_score, self.special_care_score)

    @property
    def is_nsfw(self):
        return self.nsfw_score > 0

    @property
    def is_special_care_nsfw(self):
        return self.special_care_nsfw_score > 0


class EnhancedStableDiffusionSafetyChecker(
    safety_checker_mod.StableDiffusionSafetyChecker
):
    @torch.no_grad()
    def forward(self, clip_input):
        pooled_output = self.vision_model(clip_input)[1]
        image_embeds = self.visual_projection(pooled_output)

        special_cos_dist = (
            safety_checker_mod.cosine_distance(image_embeds, self.special_care_embeds)
            .cpu()
            .numpy()
        )
        cos_dist = (
            safety_checker_mod.cosine_distance(image_embeds, self.concept_embeds)
            .cpu()
            .numpy()
        )

        safety_results = []
        batch_size = image_embeds.shape[0]
        for i in range(batch_size):
            safety_result = SafetyResult()

            for concet_idx in range(len(special_cos_dist[0])):
                concept_cos = special_cos_dist[i][concet_idx]
                concept_threshold = self.special_care_embeds_weights[concet_idx].item()
                safety_result.add_special_care_score(
                    concet_idx, concept_cos, concept_threshold
                )

            for concet_idx in range(len(cos_dist[0])):
                concept_cos = cos_dist[i][concet_idx]
                concept_threshold = self.concept_embeds_weights[concet_idx].item()
                safety_result.add_nsfw_score(concet_idx, concept_cos, concept_threshold)

            safety_results.append(safety_result)

        return safety_results


@lru_cache
def safety_models():
    safety_model_id = "CompVis/stable-diffusion-safety-checker"
    monkeypatch_safety_cosine_distance()
    safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
    safety_checker = EnhancedStableDiffusionSafetyChecker.from_pretrained(
        safety_model_id
    )
    return safety_feature_extractor, safety_checker


@lru_cache
def monkeypatch_safety_cosine_distance():
    orig_cosine_distance = safety_checker_mod.cosine_distance

    def cosine_distance_float32(image_embeds, text_embeds):
        """
        In some environments we need to distance to be in float32
        but it was coming as BFloat16.
        """
        return orig_cosine_distance(image_embeds, text_embeds).to(torch.float32)

    safety_checker_mod.cosine_distance = cosine_distance_float32


_CONCEPT_DESCRIPTIONS = []
_SPECIAL_CARE_DESCRIPTIONS = []


def create_safety_score(img, safety_mode=SafetyMode.STRICT):
    if is_blurry(img):
        sr = SafetyResult()
        sr.add_special_care_score(0, 0, 1)
        sr.add_special_care_score(1, 0, 1)
        sr.add_special_care_score(2, 0, 1)
        sr.add_nsfw_score(0, 0, 1)
        return sr

    safety_feature_extractor, safety_checker = safety_models()
    safety_checker_input = safety_feature_extractor([img], return_tensors="pt")
    clip_input = safety_checker_input.pixel_values

    safety_result = safety_checker(clip_input)[0]

    if safety_result.is_special_care_nsfw:
        img.paste((150, 0, 0), (0, 0, img.size[0], img.size[1]))
        safety_result.is_filtered = True
        logger.info(
            f"    ⚠️🔞️  Filtering NSFW image. nsfw score: {safety_result.nsfw_score}"
        )
    elif safety_mode == SafetyMode.STRICT and safety_result.is_nsfw:
        img.paste((50, 0, 0), (0, 0, img.size[0], img.size[1]))
        safety_result.is_filtered = True
        logger.info(
            f"    ⚠️  Filtering NSFW image. nsfw score: {safety_result.nsfw_score}"
        )

    return safety_result
