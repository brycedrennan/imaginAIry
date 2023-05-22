from typing import Optional

from pydantic import BaseModel

from imaginairy.http.utils import Base64Bytes


class ImagineWebPrompt(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    prompt: Optional[str]
    negative_prompt: Optional[str] = None
    prompt_strength: Optional[float] = None
    init_image: Optional[Base64Bytes] = None
    init_image_strength: Optional[float] = None
    # control_inputs: Optional[List[ControlInput]] = None
    mask_prompt: Optional[str] = None
    mask_image: Optional[Base64Bytes] = None
    mask_mode: str = "replace"
    mask_modify_original: bool = True
    outpaint: Optional[str] = None
    seed: Optional[int] = None
    steps: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    upscale: bool = False
    fix_faces: bool = False
    fix_faces_fidelity: float = 0.2
    sampler_type: Optional[str] = None
    conditioning: Optional[str] = None
    tile_mode: str = ""
    allow_compose_phase: bool = True
    model: Optional[str] = None
    model_config_path: Optional[str] = None
    is_intermediate: bool = False
    collect_progress_latents: bool = False
    caption_text: str = ""

    @classmethod
    def from_stable_studio_input(cls, stable_input):
        positive_prompt = stable_input.prompts[0].text
        negative_prompt = (
            stable_input.prompts[1].text if len(stable_input.prompts) > 1 else None
        )

        init_image = None
        init_image_strength = None
        if stable_input.initial_image:
            init_image = stable_input.initial_image.blob
            init_image_strength = stable_input.initial_image.weight

        mask_image = stable_input.mask_image.blob if stable_input.mask_image else None

        sampler_type = stable_input.sampler.id if stable_input.sampler else None

        return cls(
            prompt=positive_prompt,
            prompt_strength=stable_input.cfg_scale,
            negative_prompt=negative_prompt,
            model=stable_input.model,
            sampler_type=sampler_type,
            seed=stable_input.seed,
            steps=stable_input.steps,
            height=stable_input.height,
            width=stable_input.width,
            init_image=init_image,
            init_image_strength=init_image_strength,
            mask_image=mask_image,
            mask_mode="keep",
        )

    def to_imagine_prompt(self):
        from io import BytesIO

        from PIL import Image

        from imaginairy import ImaginePrompt

        imagine_prompt = ImaginePrompt(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            prompt_strength=self.prompt_strength,
            init_image=Image.open(BytesIO(self.init_image))
            if self.init_image
            else None,
            init_image_strength=self.init_image_strength,
            # control_inputs=self.control_inputs,  # Uncomment this if the control_inputs field exists in ImagineWebPrompt
            mask_prompt=self.mask_prompt,
            mask_image=Image.open(BytesIO(self.mask_image))
            if self.mask_image
            else None,
            mask_mode=self.mask_mode,
            mask_modify_original=self.mask_modify_original,
            outpaint=self.outpaint,
            seed=self.seed,
            steps=self.steps,
            height=self.height,
            width=self.width,
            upscale=self.upscale,
            fix_faces=self.fix_faces,
            fix_faces_fidelity=self.fix_faces_fidelity,
            sampler_type=self.sampler_type,
            conditioning=self.conditioning,
            tile_mode=self.tile_mode,
            allow_compose_phase=self.allow_compose_phase,
            model=self.model,
            model_config_path=self.model_config_path,
            is_intermediate=self.is_intermediate,
            collect_progress_latents=self.collect_progress_latents,
            caption_text=self.caption_text,
        )

        return imagine_prompt
