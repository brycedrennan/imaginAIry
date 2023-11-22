from dataclasses import dataclass

DEFAULT_MODEL = "SD-1.5"
DEFAULT_SAMPLER = "k_dpmpp_2m"

DEFAULT_NEGATIVE_PROMPT = (
    "Ugly, duplication, duplicates, mutilation, deformed, mutilated, mutation, twisted body, disfigured, bad anatomy, "
    "out of frame, extra fingers, mutated hands, "
    "poorly drawn hands, extra limbs, malformed limbs, missing arms, extra arms, missing legs, extra legs, mutated hands, "
    "extra hands, fused fingers, missing fingers, extra fingers, long neck, small head, closed eyes, rolling eyes, "
    "weird eyes, smudged face, blurred face, poorly drawn face, mutation, mutilation, cloned face, strange mouth, "
    "grainy, blurred, blurry, writing, calligraphy, signature, text, watermark, bad art,"
)

SPLITMEM_ENABLED = False


@dataclass
class ModelConfig:
    description: str
    short_name: str
    config_path: str
    weights_url: str
    default_image_size: int
    weights_url_full: str = None
    forced_attn_precision: str = "default"
    default_negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    alias: str = None


midas_url = "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt"

MODEL_CONFIGS = [
    ModelConfig(
        description="Stable Diffusion 1.5",
        short_name="SD-1.5",
        config_path="configs/stable-diffusion-v1.yaml",
        weights_url="https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/889b629140e71758e1e0006e355c331a5744b4bf/v1-5-pruned-emaonly.ckpt",
        weights_url_full="https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/889b629140e71758e1e0006e355c331a5744b4bf/v1-5-pruned.ckpt",
        default_image_size=512,
        alias="sd15",
    ),
    ModelConfig(
        description="Stable Diffusion 1.5 - Inpainting",
        short_name="SD-1.5-inpaint",
        config_path="configs/stable-diffusion-v1-inpaint.yaml",
        weights_url="https://huggingface.co/julienacquaviva/inpainting/resolve/2155ff7fe38b55f4c0d99c2f1ab9b561f8311ca7/sd-v1-5-inpainting.ckpt",
        default_image_size=512,
        alias="sd15in",
    ),
    # ModelConfig(
    #     description="Instruct Pix2Pix - Photo Editing",
    #     short_name="instruct-pix2pix",
    #     config_path="configs/instruct-pix2pix.yaml",
    #     weights_url="https://huggingface.co/imaginairy/instruct-pix2pix/resolve/ea0009b3d0d4888f410a40bd06d69516d0b5a577/instruct-pix2pix-00-22000-pruned.ckpt",
    #     default_image_size=512,
    #     default_negative_prompt="",
    #     alias="edit",
    # ),
    ModelConfig(
        description="OpenJourney V1",
        short_name="openjourney-v1",
        config_path="configs/stable-diffusion-v1.yaml",
        weights_url="https://huggingface.co/prompthero/openjourney/resolve/7428477dad893424c92f6ea1cc29d45f6d1448c1/mdjrny-v4.safetensors",
        default_image_size=512,
        default_negative_prompt="",
        alias="oj1",
    ),
    ModelConfig(
        description="OpenJourney V2",
        short_name="openjourney-v2",
        config_path="configs/stable-diffusion-v1.yaml",
        weights_url="https://huggingface.co/prompthero/openjourney-v2/resolve/47257274a40e93dab7fbc0cd2cfd5f5704cfeb60/openjourney-v2.ckpt",
        default_image_size=512,
        default_negative_prompt="",
        alias="oj2",
    ),
    ModelConfig(
        description="OpenJourney V4",
        short_name="openjourney-v4",
        config_path="configs/stable-diffusion-v1.yaml",
        weights_url="https://huggingface.co/prompthero/openjourney/resolve/e291118e93d5423dc88ac1ed93c02362b17d698f/mdjrny-v4.safetensors",
        default_image_size=512,
        default_negative_prompt="",
        alias="oj4",
    ),
]


video_models = [
    {
        "short_name": "svd",
        "description": "Stable Video Diffusion",
        "default_frames": 14,
        "default_steps": 25,
        "config_path": "configs/svd.yaml",
        "weights_url": "https://huggingface.co/imaginairy/stable-video-diffusion/resolve/f9dce2757a0713da6262f35438050357c2be7ee6/svd.fp16.safetensors",
    },
    {
        "short_name": "svd_image_decoder",
        "description": "Stable Video Diffusion - Image Decoder",
        "default_frames": 14,
        "default_steps": 25,
        "config_path": "configs/svd_image_decoder.yaml",
        "weights_url": "https://huggingface.co/imaginairy/stable-video-diffusion/resolve/f9dce2757a0713da6262f35438050357c2be7ee6/svd_image_decoder.fp16.safetensors",
    },
    {
        "short_name": "svd_xt",
        "description": "Stable Video Diffusion - XT",
        "default_frames": 25,
        "default_steps": 30,
        "config_path": "configs/svd_xt.yaml",
        "weights_url": "https://huggingface.co/imaginairy/stable-video-diffusion/resolve/f9dce2757a0713da6262f35438050357c2be7ee6/svd_xt.fp16.safetensors",
    },
    {
        "short_name": "svd_xt_image_decoder",
        "description": "Stable Video Diffusion - XT - Image Decoder",
        "default_frames": 25,
        "default_steps": 30,
        "config_path": "configs/svd_xt_image_decoder.yaml",
        "weights_url": "https://huggingface.co/imaginairy/stable-video-diffusion/resolve/f9dce2757a0713da6262f35438050357c2be7ee6/svd_xt_image_decoder.fp16.safetensors",
    },
]
video_models = {m["short_name"]: m for m in video_models}

MODEL_CONFIG_SHORTCUTS = {m.short_name: m for m in MODEL_CONFIGS}
for m in MODEL_CONFIGS:
    if m.alias:
        MODEL_CONFIG_SHORTCUTS[m.alias] = m

MODEL_CONFIG_SHORTCUTS["openjourney"] = MODEL_CONFIG_SHORTCUTS["openjourney-v2"]
MODEL_CONFIG_SHORTCUTS["oj"] = MODEL_CONFIG_SHORTCUTS["openjourney-v2"]

MODEL_SHORT_NAMES = sorted(MODEL_CONFIG_SHORTCUTS.keys())


@dataclass
class ControlNetConfig:
    short_name: str
    control_type: str
    config_path: str
    weights_url: str
    alias: str = None


CONTROLNET_CONFIGS = [
    ControlNetConfig(
        short_name="canny15",
        control_type="canny",
        config_path="configs/control-net-v15.yaml",
        weights_url="https://huggingface.co/lllyasviel/control_v11p_sd15_canny/resolve/115a470d547982438f70198e353a921996e2e819/diffusion_pytorch_model.fp16.safetensors",
        alias="canny",
    ),
    ControlNetConfig(
        short_name="depth15",
        control_type="depth",
        config_path="configs/control-net-v15.yaml",
        weights_url="https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth/resolve/539f99181d33db39cf1af2e517cd8056785f0a87/diffusion_pytorch_model.fp16.safetensors",
        alias="depth",
    ),
    ControlNetConfig(
        short_name="normal15",
        control_type="normal",
        config_path="configs/control-net-v15.yaml",
        weights_url="https://huggingface.co/lllyasviel/control_v11p_sd15_normalbae/resolve/cb7296e6587a219068e9d65864e38729cd862aa8/diffusion_pytorch_model.fp16.safetensors",
        alias="normal",
    ),
    ControlNetConfig(
        short_name="hed15",
        control_type="hed",
        config_path="configs/control-net-v15.yaml",
        weights_url="https://huggingface.co/lllyasviel/control_v11p_sd15_softedge/resolve/b5bcad0c48e9b12f091968cf5eadbb89402d6bc9/diffusion_pytorch_model.fp16.safetensors",
        alias="hed",
    ),
    ControlNetConfig(
        short_name="openpose15",
        control_type="openpose",
        config_path="configs/control-net-v15.yaml",
        weights_url="https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/9ae9f970358db89e211b87c915f9535c6686d5ba/diffusion_pytorch_model.fp16.safetensors",
        alias="openpose",
    ),
    ControlNetConfig(
        short_name="shuffle15",
        control_type="shuffle",
        config_path="configs/control-net-v15-pool.yaml",
        weights_url="https://huggingface.co/lllyasviel/control_v11e_sd15_shuffle/resolve/8cf275970f984acf5cc0fdfa537db8be098936a3/diffusion_pytorch_model.fp16.safetensors",
        alias="shuffle",
    ),
    # "instruct pix2pix"
    ControlNetConfig(
        short_name="edit15",
        control_type="edit",
        config_path="configs/control-net-v15.yaml",
        weights_url="https://huggingface.co/lllyasviel/control_v11e_sd15_ip2p/resolve/1fed6ebb905c61929a60514830eb05b039969d6d/diffusion_pytorch_model.fp16.safetensors",
        alias="edit",
    ),
    ControlNetConfig(
        short_name="inpaint15",
        control_type="inpaint",
        config_path="configs/control-net-v15.yaml",
        weights_url="https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint/resolve/c96e03a807e64135568ba8aecb66b3a306ec73bd/diffusion_pytorch_model.fp16.safetensors",
        alias="inpaint",
    ),
    ControlNetConfig(
        short_name="details15",
        control_type="details",
        config_path="configs/control-net-v15.yaml",
        weights_url="https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile/resolve/3f877705c37010b7221c3d10743307d6b5b6efac/diffusion_pytorch_model.bin",
        alias="details",
    ),
    ControlNetConfig(
        short_name="colorize15",
        control_type="colorize",
        config_path="configs/control-net-v15.yaml",
        weights_url="https://huggingface.co/ioclab/control_v1p_sd15_brightness/resolve/8509361eb1ba89c03839040ed8c75e5f11bbd9c5/diffusion_pytorch_model.safetensors",
        alias="colorize",
    ),
]

CONTROLNET_CONFIG_SHORTCUTS = {}
for m in CONTROLNET_CONFIGS:
    if m.alias:
        CONTROLNET_CONFIG_SHORTCUTS[m.alias] = m

for m in CONTROLNET_CONFIGS:
    CONTROLNET_CONFIG_SHORTCUTS[m.short_name] = m

SAMPLER_TYPE_OPTIONS = [
    # "plms",
    "ddim",
    "k_dpmpp_2m"
    # "k_dpm_fast",
    # "k_dpm_adaptive",
    # "k_lms",
    # "k_dpm_2",
    # "k_dpm_2_a",
    # "k_dpmpp_2m",
    # "k_dpmpp_2s_a",
    # "k_euler",
    # "k_euler_a",
    # "k_heun",
]
