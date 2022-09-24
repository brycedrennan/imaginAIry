import kornia
import torch
from einops import repeat
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer

from imaginairy.utils import get_device
from imaginairy.vendored import clip


class FrozenCLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        device=get_device(),
        max_length=77,
    ):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """

    def __init__(
        self,
        version="ViT-L/14",
        device=get_device(),
        max_length=77,
        n_repeat=1,
        normalize=True,
    ):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device=device)
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim == 2:
            z = z[:, None, :]
        z = repeat(z, "b 1 d -> b k d", k=self.n_repeat)
        return z


class FrozenClipImageEmbedder(nn.Module):
    """
    Uses the CLIP image encoder.
    """

    def __init__(
        self,
        model_name,
        jit=False,
        device=get_device(),
        antialias=False,
    ):
        super().__init__()
        self.model, preprocess = clip.load(  # noqa
            name=model_name, device=device, jit=jit
        )

        self.antialias = antialias

        self.register_buffer(
            "mean", torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False
        )
        self.register_buffer(
            "std", torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False
        )

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(
            x,
            (224, 224),
            interpolation="bicubic",
            align_corners=True,
            antialias=self.antialias,
        )
        x = (x + 1.0) / 2.0
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x))
