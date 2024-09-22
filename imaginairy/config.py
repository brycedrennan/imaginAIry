"""Classes and constants for AI model configuration"""

from dataclasses import dataclass
from typing import Any, List

DEFAULT_MODEL_WEIGHTS = "sd15"
DEFAULT_SOLVER = "ddim"
DEFAULT_UPSCALE_MODEL = "realesrgan-x2-plus"

DEFAULT_NEGATIVE_PROMPT = (
    "Ugly, duplication, duplicates, mutilation, deformed, mutilated, mutation, twisted body, disfigured, bad anatomy, "
    "out of frame, extra fingers, mutated hands, "
    "poorly drawn hands, extra limbs, malformed limbs, missing arms, extra arms, missing legs, extra legs, mutated hands, "
    "extra hands, fused fingers, missing fingers, extra fingers, long neck, small head, closed eyes, rolling eyes, "
    "weird eyes, smudged face, blurred face, poorly drawn face, mutation, mutilation, cloned face, strange mouth, "
    "grainy, blurred, blurry, writing, calligraphy, signature, text, watermark, bad art,"
)


@dataclass
class ModelArchitecture:
    name: str
    aliases: List[str]
    output_modality: str
    defaults: dict[str, Any]
    config_path: str | None = None

    @property
    def primary_alias(self):
        if self.aliases:
            return self.aliases[0]


MODEL_ARCHITECTURES = [
    ModelArchitecture(
        name="Stable Diffusion 1.5",
        aliases=["sd15", "sd-15", "sd1.5", "sd-1.5"],
        output_modality="image",
        defaults={"size": "512"},
        config_path="configs/stable-diffusion-v1.yaml",
    ),
    ModelArchitecture(
        name="Stable Diffusion 1.5 - Inpainting",
        aliases=[
            "sd15inpaint",
            "sd15-inpaint",
            "sd-15-inpaint",
            "sd1.5inpaint",
            "sd1.5-inpaint",
            "sd-1.5-inpaint",
        ],
        output_modality="image",
        defaults={"size": "512"},
        config_path="configs/stable-diffusion-v1-inpaint.yaml",
    ),
    ModelArchitecture(
        name="Stable Diffusion XL",
        aliases=["sdxl", "sd-xl"],
        output_modality="image",
        defaults={"size": "1024"},
    ),
    ModelArchitecture(
        name="Stable Diffusion XL - Inpainting",
        aliases=["sdxlinpaint", "sd-xlinpaint", "sdxl-inpaint"],
        output_modality="image",
        defaults={"size": "1024"},
    ),
    ModelArchitecture(
        name="Stable Video Diffusion",
        aliases=["svd", "stablevideo"],
        output_modality="video",
        defaults={"size": "1024x576"},
        config_path="configs/svd.yaml",
    ),
    ModelArchitecture(
        name="Stable Video Diffusion - Image Decoder",
        aliases=["svd-image-decoder", "svd-imdec"],
        output_modality="video",
        defaults={"size": "1024x576"},
        config_path="configs/svd_image_decoder.yaml",
    ),
    ModelArchitecture(
        name="Stable Video Diffusion - XT",
        aliases=["svd-xt", "svd25f", "svd-25f", "stablevideoxt", "svdxt"],
        output_modality="video",
        defaults={"size": "1024x576"},
        config_path="configs/svd_xt.yaml",
    ),
    ModelArchitecture(
        name="Stable Video Diffusion - XT - Image Decoder",
        aliases=[
            "svd-xt-image-decoder",
            "svd-xt-imdec",
            "svd-25f-imdec",
            "svdxt-imdec",
            "svdxtimdec",
            "svd25fimdec",
            "svdxtimdec",
        ],
        output_modality="video",
        defaults={"size": "1024x576"},
        config_path="configs/svd_xt_image_decoder.yaml",
    ),
    ModelArchitecture(
        name="Flux",
        aliases=["flux"],
        output_modality="image",
        defaults={"size": "1024", "steps": 3},
    ),
]

MODEL_ARCHITECTURE_LOOKUP = {}
for m in MODEL_ARCHITECTURES:
    for a in m.aliases:
        MODEL_ARCHITECTURE_LOOKUP[a] = m


@dataclass
class ModelWeightsConfig:
    name: str
    aliases: List[str]
    architecture: ModelArchitecture
    defaults: dict[str, Any]
    weights_location: str

    def __post_init__(self):
        if isinstance(self.architecture, str):
            self.architecture = MODEL_ARCHITECTURE_LOOKUP[self.architecture]
        if not isinstance(self.architecture, ModelArchitecture):
            msg = f"zYou must specify an architecture {self.architecture}"
            raise ValueError(msg)  # noqa


MODEL_WEIGHT_CONFIGS = [
    ModelWeightsConfig(
        name="Stable Diffusion 1.5",
        aliases=MODEL_ARCHITECTURE_LOOKUP["sd15"].aliases,
        architecture=MODEL_ARCHITECTURE_LOOKUP["sd15"],
        defaults={"negative_prompt": DEFAULT_NEGATIVE_PROMPT},
        weights_location="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/f03de327dd89b501a01da37fc5240cf4fdba85a1/",
    ),
    ModelWeightsConfig(
        name="Stable Diffusion 1.5 - Inpainting",
        aliases=MODEL_ARCHITECTURE_LOOKUP["sd15inpaint"].aliases,
        architecture=MODEL_ARCHITECTURE_LOOKUP["sd15inpaint"],
        defaults={"negative_prompt": DEFAULT_NEGATIVE_PROMPT},
        weights_location="https://huggingface.co/julienacquaviva/inpainting/resolve/2155ff7fe38b55f4c0d99c2f1ab9b561f8311ca7/sd-v1-5-inpainting.ckpt",
    ),
    ModelWeightsConfig(
        name="OpenJourney V1",
        aliases=["openjourney-v1", "oj1", "ojv1", "openjourney1"],
        architecture=MODEL_ARCHITECTURE_LOOKUP["sd15"],
        defaults={"negative_prompt": "poor quality"},
        weights_location="https://huggingface.co/prompthero/openjourney/resolve/7428477dad893424c92f6ea1cc29d45f6d1448c1/mdjrny-v4.safetensors",
    ),
    ModelWeightsConfig(
        name="OpenJourney V2",
        aliases=["openjourney-v2", "oj2", "ojv2", "openjourney2"],
        architecture=MODEL_ARCHITECTURE_LOOKUP["sd15"],
        weights_location="https://huggingface.co/prompthero/openjourney-v2/resolve/47257274a40e93dab7fbc0cd2cfd5f5704cfeb60/openjourney-v2.ckpt",
        defaults={"negative_prompt": "poor quality"},
    ),
    ModelWeightsConfig(
        name="OpenJourney V4",
        aliases=["openjourney-v4", "oj4", "ojv4", "openjourney4", "openjourney", "oj"],
        architecture=MODEL_ARCHITECTURE_LOOKUP["sd15"],
        weights_location="https://huggingface.co/prompthero/openjourney/tree/f4572661b028c732b2b97c8fbdc32fa5db3afe03/",
        defaults={"negative_prompt": "poor quality"},
    ),
    ModelWeightsConfig(
        name="Modern Disney",
        aliases=["modern-disney", "modi", "modi15", "modern-disney-15"],
        architecture=MODEL_ARCHITECTURE_LOOKUP["sd15"],
        weights_location="https://huggingface.co/nitrosocke/mo-di-diffusion/tree/e3106d24aa8c37bf856257daea2ae789eabc4d70/",
        defaults={"negative_prompt": DEFAULT_NEGATIVE_PROMPT},
    ),
    ModelWeightsConfig(
        name="Miniaturus Potentia V1.2",
        aliases=[
            "miniaturuspotentia",
            "potentia",
            "miniaturuspotentia12",
            "mp12",
            "mp",
            "potentia12",
        ],
        architecture=MODEL_ARCHITECTURE_LOOKUP["sd15"],
        weights_location="https://huggingface.co/dataautogpt3/Miniaturus_PotentiaV1.2/tree/7ef539518ad5ad591c45f0b920050883f7e51e83/",
        defaults={"negative_prompt": DEFAULT_NEGATIVE_PROMPT},
    ),
    # SDXL Weights
    ModelWeightsConfig(
        name="Stable Diffusion XL",
        aliases=MODEL_ARCHITECTURE_LOOKUP["sdxl"].aliases,
        architecture=MODEL_ARCHITECTURE_LOOKUP["sdxl"],
        defaults={
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
            "composition_strength": 0.6,
        },
        weights_location="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/462165984030d82259a11f4367a4eed129e94a7b/",
    ),
    ModelWeightsConfig(
        name="Stable Diffusion XL - Inpainting",
        aliases=MODEL_ARCHITECTURE_LOOKUP["sdxl-inpaint"].aliases,
        architecture=MODEL_ARCHITECTURE_LOOKUP["sdxl-inpaint"],
        defaults={
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
            "composition_strength": 0.6,
        },
        weights_location="https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/tree/115134f363124c53c7d878647567d04daf26e41e/",
    ),
    ModelWeightsConfig(
        name="OpenDalle V1.1",
        aliases=["opendalle11", "odv11", "opendalle11", "opendalle", "od"],
        architecture=MODEL_ARCHITECTURE_LOOKUP["sdxl"],
        defaults={
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
            "composition_strength": 0.6,
        },
        weights_location="https://huggingface.co/dataautogpt3/OpenDalleV1.1/tree/33dc6acd722cd7a956bf676011609e41665d4c4e/",
    ),
    # Video Weights
    ModelWeightsConfig(
        name="Stable Video Diffusion",
        aliases=MODEL_ARCHITECTURE_LOOKUP["svd"].aliases,
        architecture=MODEL_ARCHITECTURE_LOOKUP["svd"],
        weights_location="https://huggingface.co/imaginairy/stable-video-diffusion/resolve/f9dce2757a0713da6262f35438050357c2be7ee6/svd.fp16.safetensors",
        defaults={"frames": 14, "steps": 25},
    ),
    ModelWeightsConfig(
        name="Stable Video Diffusion - Image Decoder",
        aliases=MODEL_ARCHITECTURE_LOOKUP["svd-image-decoder"].aliases,
        architecture=MODEL_ARCHITECTURE_LOOKUP["svd-image-decoder"],
        weights_location="https://huggingface.co/imaginairy/stable-video-diffusion/resolve/f9dce2757a0713da6262f35438050357c2be7ee6/svd_image_decoder.fp16.safetensors",
        defaults={"frames": 14, "steps": 25},
    ),
    ModelWeightsConfig(
        name="Stable Video Diffusion - XT",
        aliases=MODEL_ARCHITECTURE_LOOKUP["svdxt"].aliases,
        architecture=MODEL_ARCHITECTURE_LOOKUP["svdxt"],
        weights_location="https://huggingface.co/imaginairy/stable-video-diffusion/resolve/f9dce2757a0713da6262f35438050357c2be7ee6/svd_xt.fp16.safetensors",
        defaults={"frames": 25, "steps": 30},
    ),
    ModelWeightsConfig(
        name="Stable Video Diffusion - XT - Image Decoder",
        aliases=MODEL_ARCHITECTURE_LOOKUP["svd-xt-image-decoder"].aliases,
        architecture=MODEL_ARCHITECTURE_LOOKUP["svd-xt-image-decoder"],
        weights_location="https://huggingface.co/imaginairy/stable-video-diffusion/resolve/f9dce2757a0713da6262f35438050357c2be7ee6/svd_xt_image_decoder.fp16.safetensors",
        defaults={"frames": 25, "steps": 30},
    ),
    ModelWeightsConfig(
        name="FLUX.1-schnell",
        aliases=MODEL_ARCHITECTURE_LOOKUP["flux"].aliases,
        architecture=MODEL_ARCHITECTURE_LOOKUP["flux"],
        weights_location="",
        defaults={
            "steps": 5,
            "negative_prompt": "",
        },
    ),
]

MODEL_WEIGHT_CONFIG_LOOKUP = {}
for mw in MODEL_WEIGHT_CONFIGS:
    for a in mw.aliases:
        MODEL_WEIGHT_CONFIG_LOOKUP[a] = mw


IMAGE_WEIGHTS_SHORT_NAMES = [
    k
    for k, mw in MODEL_WEIGHT_CONFIG_LOOKUP.items()
    if mw.architecture.output_modality == "image"
]
IMAGE_WEIGHTS_SHORT_NAMES.sort()


@dataclass
class ControlConfig:
    name: str
    aliases: List[str]
    control_type: str
    config_path: str
    weights_location: str


CONTROL_CONFIGS = [
    ControlConfig(
        name="Canny Edge Control",
        aliases=["canny", "canny15"],
        control_type="canny",
        config_path="configs/control-net-v15.yaml",
        weights_location="https://huggingface.co/lllyasviel/control_v11p_sd15_canny/resolve/115a470d547982438f70198e353a921996e2e819/diffusion_pytorch_model.fp16.safetensors",
    ),
    ControlConfig(
        name="Depth Control",
        aliases=["depth", "depth15"],
        control_type="depth",
        config_path="configs/control-net-v15.yaml",
        weights_location="https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth/resolve/539f99181d33db39cf1af2e517cd8056785f0a87/diffusion_pytorch_model.fp16.safetensors",
    ),
    ControlConfig(
        name="Normal Map Control",
        aliases=["normal", "normal15"],
        control_type="normal",
        config_path="configs/control-net-v15.yaml",
        weights_location="https://huggingface.co/lllyasviel/control_v11p_sd15_normalbae/resolve/cb7296e6587a219068e9d65864e38729cd862aa8/diffusion_pytorch_model.fp16.safetensors",
    ),
    ControlConfig(
        name="Soft Edge Control (HED)",
        aliases=["hed", "hed15"],
        control_type="hed",
        config_path="configs/control-net-v15.yaml",
        weights_location="https://huggingface.co/lllyasviel/control_v11p_sd15_softedge/resolve/b5bcad0c48e9b12f091968cf5eadbb89402d6bc9/diffusion_pytorch_model.fp16.safetensors",
    ),
    ControlConfig(
        name="Pose Control",
        control_type="openpose",
        config_path="configs/control-net-v15.yaml",
        weights_location="https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/9ae9f970358db89e211b87c915f9535c6686d5ba/diffusion_pytorch_model.fp16.safetensors",
        aliases=["openpose", "pose", "pose15", "openpose15"],
    ),
    ControlConfig(
        name="Shuffle Control",
        control_type="shuffle",
        config_path="configs/control-net-v15-pool.yaml",
        weights_location="https://huggingface.co/lllyasviel/control_v11e_sd15_shuffle/resolve/8cf275970f984acf5cc0fdfa537db8be098936a3/diffusion_pytorch_model.fp16.safetensors",
        aliases=["shuffle", "shuffle15"],
    ),
    # "instruct pix2pix"
    ControlConfig(
        name="Edit Prompt Control",
        aliases=["edit", "edit15"],
        control_type="edit",
        config_path="configs/control-net-v15.yaml",
        weights_location="https://huggingface.co/lllyasviel/control_v11e_sd15_ip2p/resolve/1fed6ebb905c61929a60514830eb05b039969d6d/diffusion_pytorch_model.fp16.safetensors",
    ),
    ControlConfig(
        name="Inpaint Control",
        aliases=["inpaint", "inpaint15"],
        control_type="inpaint",
        config_path="configs/control-net-v15.yaml",
        weights_location="https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint/resolve/c96e03a807e64135568ba8aecb66b3a306ec73bd/diffusion_pytorch_model.fp16.safetensors",
    ),
    ControlConfig(
        name="Details Control (Upscale Tile)",
        aliases=["details", "details15"],
        control_type="details",
        config_path="configs/control-net-v15.yaml",
        weights_location="https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile/resolve/3f877705c37010b7221c3d10743307d6b5b6efac/diffusion_pytorch_model.bin",
    ),
    ControlConfig(
        name="Brightness Control (Colorize)",
        aliases=["colorize", "colorize15"],
        control_type="colorize",
        config_path="configs/control-net-v15.yaml",
        weights_location="https://huggingface.co/ioclab/control_v1p_sd15_brightness/resolve/8509361eb1ba89c03839040ed8c75e5f11bbd9c5/diffusion_pytorch_model.safetensors",
    ),
    ControlConfig(
        name="qrcode",
        control_type="qrcode",
        config_path="configs/control-net-v15.yaml",
        weights_location="https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster/resolve/4a946e610f670c4cd6cf46b8641fca190e4f56c4/diffusion_pytorch_model.safetensors",
        aliases=["qrcode"],
    ),
    ControlConfig(
        name="DensePose",
        control_type="densepose",
        config_path="configs/control-net-v15.yaml",
        weights_location="https://huggingface.co/zcxu-eric/MagicAnimate/resolve/3d80ae8c50b289e55ee68deecc83afaab9c6a382/densepose_controlnet/diffusion_pytorch_model.safetensors?download=true",
        aliases=["densepose"],
    ),
]

CONTROL_CONFIG_SHORTCUTS: dict[str, ControlConfig] = {}
for cc in CONTROL_CONFIGS:
    for ca in cc.aliases:
        CONTROL_CONFIG_SHORTCUTS[ca] = cc


@dataclass
class SolverConfig:
    name: str
    short_name: str
    aliases: List[str]
    papers: List[str]
    implementations: List[str]


SOLVER_CONFIGS = [
    SolverConfig(
        name="DDIM",
        short_name="DDIM",
        aliases=["ddim"],
        papers=["https://arxiv.org/abs/2010.02502"],
        implementations=[
            "https://github.com/ermongroup/ddim",
            "https://github.com/Stability-AI/stablediffusion/blob/cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf/ldm/models/diffusion/ddim.py#L10",
            "https://github.com/huggingface/diffusers/blob/76c645d3a641c879384afcb43496f0b7db8cc5cb/src/diffusers/schedulers/scheduling_ddim.py#L131",
        ],
    ),
    SolverConfig(
        name="DPM-Solver++",
        short_name="DPMPP",
        aliases=["dpmpp", "dpm++", "dpmsolver"],
        papers=["https://arxiv.org/abs/2211.01095"],
        implementations=[
            "https://github.com/LuChengTHU/dpm-solver/blob/52bc3fbcd5de56d60917b826b15d2b69460fc2fa/dpm_solver_pytorch.py#L337",
            "https://github.com/apple/ml-stable-diffusion/blob/7449ce46a4b23c94413b714704202e4ea4c55080/swift/StableDiffusion/pipeline/DPMSolverMultistepScheduler.swift#L27",
            "https://github.com/crowsonkb/k-diffusion/blob/045515774882014cc14c1ba2668ab5bad9cbf7c0/k_diffusion/sampling.py#L509",
        ],
    ),
]

_ip_adapter_commit = "92a2d51861c754afacf8b3aaf90845254b49f219"
IP_ADAPTER_WEIGHT_LOCATIONS = {
    "sd15": {
        "full-face": f"https://huggingface.co/h94/IP-Adapter/resolve/{_ip_adapter_commit}/models/ip-adapter-full-face_sd15.safetensors",
        "plus-face": f"https://huggingface.co/h94/IP-Adapter/resolve/{_ip_adapter_commit}/models/ip-adapter-plus-face_sd15.safetensors",
        "plus": f"https://huggingface.co/h94/IP-Adapter/resolve/{_ip_adapter_commit}/models/ip-adapter-plus_sd15.safetensors",
        "normal": f"https://huggingface.co/h94/IP-Adapter/resolve/{_ip_adapter_commit}/models/ip-adapter_sd15.safetensors",
        "light": f"https://huggingface.co/h94/IP-Adapter/resolve/{_ip_adapter_commit}/models/ip-adapter_sd15_light.safetensors",
        "vitg": f"https://huggingface.co/h94/IP-Adapter/resolve/{_ip_adapter_commit}/models/ip-adapter_sd15_vit-G.safetensors",
    },
    "sdxl": {
        "plus-face": f"https://huggingface.co/h94/IP-Adapter/blob/{_ip_adapter_commit}/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors",
        "plus": f"https://huggingface.co/h94/IP-Adapter/resolve/{_ip_adapter_commit}/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors",
        "vit-g": f"https://huggingface.co/h94/IP-Adapter/resolve/{_ip_adapter_commit}/sdxl_models/ip-adapter_sdxl.safetensors",
        "normal": f"https://huggingface.co/h94/IP-Adapter/resolve/{_ip_adapter_commit}/sdxl_models/ip-adapter_sdxl_vit-h.safetensors",
    },
}
SD21_UNCLIP_WEIGHTS_URL = "https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip/resolve/e99f66a92bdcd1b0fb0d4b6a9b81b3b37d8bea44/image_encoder/model.fp16.safetensors"
DENSEPOSE_REPO_URL = "https://huggingface.co/LayerNorm/DensePose-TorchScript-with-hint-image/resolve/65446422ea6225b9d72f93f3d2e2ad55e78b0b78"

SOLVER_TYPE_NAMES = [s.aliases[0] for s in SOLVER_CONFIGS]

SOLVER_LOOKUP = {}
for s in SOLVER_CONFIGS:
    for a in s.aliases:
        SOLVER_LOOKUP[a.lower()] = s
