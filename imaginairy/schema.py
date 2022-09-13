import hashlib
import json
import random
from datetime import datetime, timezone

import numpy
from PIL.Image import Exif

from imaginairy.utils import get_device, get_device_name


class WeightedPrompt:
    def __init__(self, text, weight=1):
        self.text = text
        self.weight = weight

    def __str__(self):
        return f"{self.weight}*({self.text})"


class ImaginePrompt:
    def __init__(
        self,
        prompt=None,
        prompt_strength=7.5,
        init_image=None,
        init_image_strength=0.3,
        seed=None,
        steps=50,
        height=512,
        width=512,
        upscale=False,
        fix_faces=False,
        sampler_type="PLMS",
    ):
        prompt = prompt if prompt is not None else "a scenic landscape"
        if isinstance(prompt, str):
            self.prompts = [WeightedPrompt(prompt, 1)]
        else:
            self.prompts = prompt
        self.prompts.sort(key=lambda p: p.weight, reverse=True)
        self.prompt_strength = prompt_strength
        self.init_image = init_image
        self.init_image_strength = init_image_strength
        self.seed = random.randint(1, 1_000_000_000) if seed is None else seed
        self.steps = steps
        self.height = height
        self.width = width
        self.upscale = upscale
        self.fix_faces = fix_faces
        self.sampler_type = sampler_type

    @property
    def prompt_text(self):
        if len(self.prompts) == 1:
            return self.prompts[0].text
        return "|".join(str(p) for p in self.prompts)

    def prompt_description(self):
        return (
            f'🖼  : "{self.prompt_text}" {self.width}x{self.height}px '
            f"seed:{self.seed} prompt-strength:{self.prompt_strength} steps:{self.steps} sampler-type:{self.sampler_type}"
        )

    def as_dict(self):
        prompts = [(p.weight, p.text) for p in self.prompts]
        return {
            "software": "imaginairy",
            "prompts": prompts,
            "prompt_strength": self.prompt_strength,
            "init_image": self.init_image,
            "init_image_strength": self.init_image_strength,
            "seed": self.seed,
            "steps": self.steps,
            "height": self.height,
            "width": self.width,
            "upscale": self.upscale,
            "fix_faces": self.fix_faces,
            "sampler_type": self.sampler_type,
        }


class ExifCodes:
    """https://www.awaresystems.be/imaging/tiff/tifftags/baseline.html"""

    ImageDescription = 0x010E
    Software = 0x0131
    DateTime = 0x0132
    HostComputer = 0x013C
    UserComment = 0x9286


class ImagineResult:
    def __init__(self, img, prompt: ImaginePrompt, upscaled_img=None):
        self.img = img
        self.upscaled_img = upscaled_img
        self.prompt = prompt
        self.created_at = datetime.utcnow().replace(tzinfo=timezone.utc)
        self.torch_backend = get_device()
        self.hardware_name = get_device_name(get_device())

    def cv2_img(self):
        open_cv_image = numpy.array(self.img)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        return open_cv_image
        # return cv2.cvtColor(numpy.array(self.img), cv2.COLOR_RGB2BGR)

    def md5(self):
        return hashlib.md5(self.img.tobytes()).hexdigest()

    def metadata_dict(self):
        return {
            "prompt": self.prompt.as_dict(),
        }

    def _exif(self):
        exif = Exif()
        exif[ExifCodes.ImageDescription] = self.prompt.prompt_description()
        exif[ExifCodes.UserComment] = json.dumps(self.metadata_dict())
        # help future web scrapes not ingest AI generated art
        exif[ExifCodes.Software] = "Imaginairy / Stable Diffusion v1.4"
        exif[ExifCodes.DateTime] = self.created_at.isoformat(sep=" ")[:19]
        exif[ExifCodes.HostComputer] = f"{self.torch_backend}:{self.hardware_name}"
        return exif

    def save(self, save_path):
        self.img.save(save_path, exif=self._exif())

    def save_upscaled(self, save_path):
        self.upscaled_img.save(save_path, exif=self._exif())
