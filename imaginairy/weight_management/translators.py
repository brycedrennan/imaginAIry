import os
from functools import lru_cache

from torch import device as Device

from imaginairy.weight_management.translation import TensorDict, WeightTranslationMap

_current_folder = os.path.dirname(os.path.abspath(__file__))
weight_map_folder = os.path.join(_current_folder, "weight_maps")


@lru_cache
def _weight_map(map_name: str) -> WeightTranslationMap:
    map_path = os.path.join(weight_map_folder, f"{map_name}.weightmap.json")
    return WeightTranslationMap.load(map_path)


def transformers_text_encoder_to_refiners_translator() -> WeightTranslationMap:
    return _weight_map("Transformers-ClipTextEncoder")


def transformers_image_encoder_to_refiners_translator() -> WeightTranslationMap:
    return _weight_map("Transformers-ClipImageEncoder-SD21")


def diffusers_autoencoder_kl_to_refiners_translator() -> WeightTranslationMap:
    return _weight_map("Diffusers-AutoencoderKL-SD")


def diffusers_unet_sd15_to_refiners_translator() -> WeightTranslationMap:
    return _weight_map("Diffusers-UNet-SD15")


def diffusers_unet_sdxl_to_refiners_translator() -> WeightTranslationMap:
    return _weight_map("Diffusers-UNet-SDXL")


def informative_drawings_to_refiners_translator() -> WeightTranslationMap:
    return _weight_map("InformativeDrawings")


def diffusers_controlnet_sd15_to_refiners_translator() -> WeightTranslationMap:
    return _weight_map("Diffusers-Controlnet-SD15")


def diffusers_ip_adapter_sd15_to_refiners_translator() -> WeightTranslationMap:
    return _weight_map("Diffusers-IPAdapter-SD15")


def diffusers_ip_adapter_sdxl_to_refiners_translator() -> WeightTranslationMap:
    return _weight_map("Diffusers-IPAdapter-SDXL")


def diffusers_ip_adapter_plus_sd15_to_refiners_translator() -> WeightTranslationMap:
    return _weight_map("Diffusers-IPAdapterPlus-SD15")


def diffusers_ip_adapter_plus_sdxl_to_refiners_translator() -> WeightTranslationMap:
    return _weight_map("Diffusers-IPAdapterPlus-SDXL")


def diffusers_t2i_adapter_sd15_to_refiners_translator() -> WeightTranslationMap:
    return _weight_map("Diffusers-T2IAdapter-SD15")


def diffusers_t2i_adapter_sdxl_to_refiners_translator() -> WeightTranslationMap:
    return _weight_map("Diffusers-T2IAdapter-SDXL")


class DoubleTextEncoderTranslator:
    def __init__(self):
        self.translator = transformers_text_encoder_to_refiners_translator()

    def load_and_translate_weights(
        self,
        text_encoder_l_weights_path: str,
        text_encoder_g_weights_path: str,
        device: Device | str = "cpu",
    ) -> TensorDict:
        text_encoder_l_weights = self.translator.load_and_translate_weights(
            text_encoder_l_weights_path, device=device
        )
        text_encoder_g_weights = self.translator.load_and_translate_weights(
            text_encoder_g_weights_path, device=device
        )
        return self.translate_weights(text_encoder_l_weights, text_encoder_g_weights)

    def translate_weights(
        self, text_encoder_l_weights: TensorDict, text_encoder_g_weights: TensorDict
    ) -> TensorDict:
        new_sd: TensorDict = {}

        for k in list(text_encoder_l_weights.keys()):
            if k.startswith("TransformerLayer_12"):
                text_encoder_l_weights.pop(k)
            elif k.startswith("LayerNorm"):
                text_encoder_l_weights.pop(k)
            else:
                new_key = f"Parallel.CLIPTextEncoderL.{k}"
                new_sd[new_key] = text_encoder_l_weights.pop(k)

        new_sd[
            "Parallel.TextEncoderWithPooling.Parallel.Chain.Linear.weight"
        ] = text_encoder_g_weights.pop("Linear.weight")
        for k in list(text_encoder_g_weights.keys()):
            if k.startswith("TransformerLayer_32"):
                new_key = f"Parallel.TextEncoderWithPooling.Parallel.Chain.CLIPTextEncoderG.TransformerLayer{k[19:]}"
            elif k.startswith("LayerNorm"):
                new_key = f"Parallel.TextEncoderWithPooling.Parallel.Chain.CLIPTextEncoderG.{k}"
            else:
                new_key = f"Parallel.TextEncoderWithPooling.CLIPTextEncoderG.{k}"

            new_sd[new_key] = text_encoder_g_weights.pop(k)

        return new_sd
