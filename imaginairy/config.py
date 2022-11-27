from dataclasses import dataclass

DEFAULT_MODEL = "SD-1.5"
DEFAULT_SAMPLER = "k_dpmpp_2m"


@dataclass
class ModelConfig:
    short_name: str
    config_path: str
    weights_url: str
    default_image_size: int


MODEL_CONFIGS = [
    ModelConfig(
        short_name="SD-1.4",
        config_path="configs/stable-diffusion-v1.yaml",
        weights_url="https://huggingface.co/bstddev/sd-v1-4/resolve/77221977fa8de8ab8f36fac0374c120bd5b53287/sd-v1-4.ckpt",
        default_image_size=512,
    ),
    ModelConfig(
        short_name="SD-1.5",
        config_path="configs/stable-diffusion-v1.yaml",
        weights_url="https://huggingface.co/acheong08/SD-V1-5-cloned/resolve/fc392f6bd4345b80fc2256fa8aded8766b6c629e/v1-5-pruned-emaonly.ckpt",
        default_image_size=512,
    ),
    ModelConfig(
        short_name="SD-1.5-inpaint",
        config_path="configs/stable-diffusion-v1-inpaint.yaml",
        weights_url="https://huggingface.co/julienacquaviva/inpainting/resolve/2155ff7fe38b55f4c0d99c2f1ab9b561f8311ca7/sd-v1-5-inpainting.ckpt",
        default_image_size=512,
    ),
    ModelConfig(
        short_name="SD-2.0",
        config_path="configs/stable-diffusion-v2-inference.yaml",
        weights_url="https://huggingface.co/stabilityai/stable-diffusion-2-base/resolve/main/512-base-ema.ckpt",
        default_image_size=512,
    ),
    ModelConfig(
        short_name="SD-2.0-inpaint",
        config_path="configs/stable-diffusion-v2-inpainting-inference.yaml",
        weights_url="https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/resolve/main/512-inpainting-ema.ckpt",
        default_image_size=512,
    ),
    ModelConfig(
        short_name="SD-2.0-v",
        config_path="configs/stable-diffusion-v2-inference-v.yaml",
        weights_url="https://huggingface.co/stabilityai/stable-diffusion-2/resolve/main/768-v-ema.ckpt",
        default_image_size=768,
    ),
    ModelConfig(
        short_name="SD-2.0-upscale",
        config_path="configs/stable-diffusion-v2-upscaling.yaml",
        weights_url="https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/resolve/main/x4-upscaler-ema.ckpt",
        default_image_size=512,
    ),
]

MODEL_CONFIG_SHORTCUTS = {m.short_name: m for m in MODEL_CONFIGS}
