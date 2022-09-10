import hashlib
import random

import numpy


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
        seed=None,
        prompt_strength=7.5,
        sampler_type="PLMS",
        init_image=None,
        init_image_strength=0.3,
        steps=50,
        height=512,
        width=512,
        upscale=False,
        fix_faces=False,
        parts=None,
    ):
        prompt = prompt if prompt is not None else "a scenic landscape"
        if isinstance(prompt, str):
            self.prompts = [WeightedPrompt(prompt, 1)]
        else:
            self.prompts = prompt
        self.init_image = init_image
        self.init_image_strength = init_image_strength
        self.prompts.sort(key=lambda p: p.weight, reverse=True)
        self.seed = random.randint(1, 1_000_000_000) if seed is None else seed
        self.prompt_strength = prompt_strength
        self.sampler_type = sampler_type
        self.steps = steps
        self.height = height
        self.width = width
        self.upscale = upscale
        self.fix_faces = fix_faces
        self.parts = parts or {}

    @property
    def prompt_text(self):
        if len(self.prompts) == 1:
            return self.prompts[0].text
        return "|".join(str(p) for p in self.prompts)


class ImagineResult:
    def __init__(self, img, prompt):
        self.img = img
        self.prompt = prompt

    def cv2_img(self):
        open_cv_image = numpy.array(self.img)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        return open_cv_image
        # return cv2.cvtColor(numpy.array(self.img), cv2.COLOR_RGB2BGR)

    def md5(self):
        return hashlib.md5(self.img.tobytes()).hexdigest()
