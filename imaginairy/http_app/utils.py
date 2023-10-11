import base64
from io import BytesIO

from imaginairy import imagine


def generate_image(prompt):
    """ImaginePrompt to generated image."""
    result = next(imagine([prompt]))
    img = result.images["generated"]
    img_io = BytesIO()
    img.save(img_io, "JPEG")
    img_io.seek(0)
    return img_io


def generate_image_b64(prompt):
    """ImaginePrompt to generated base64 encoded image."""
    img_io = generate_image(prompt)
    img_base64 = base64.b64encode(img_io.getvalue())
    return img_base64


class Base64Bytes(bytes):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, info):
        if isinstance(v, bytes):
            return v
        if isinstance(v, str):
            return base64.b64decode(v)
        raise ValueError("Byte value must be either str or bytes")

    def __str__(self):
        return base64.b64encode(self).decode()
