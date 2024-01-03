from typing import cast

from jaxtyping import Float
from torch import Tensor, cat, device as Device, dtype as DType

import imaginairy.vendored.refiners.fluxion.layers as fl
from imaginairy.vendored.refiners.fluxion.adapters.adapter import Adapter
from imaginairy.vendored.refiners.fluxion.context import Contexts
from imaginairy.vendored.refiners.foundationals.clip.text_encoder import CLIPTextEncoderG, CLIPTextEncoderL
from imaginairy.vendored.refiners.foundationals.clip.tokenizer import CLIPTokenizer


class TextEncoderWithPooling(fl.Chain, Adapter[CLIPTextEncoderG]):
    def __init__(
        self,
        target: CLIPTextEncoderG,
        projection: fl.Linear | None = None,
    ) -> None:
        with self.setup_adapter(target=target):
            tokenizer = target.ensure_find(CLIPTokenizer)
            super().__init__(
                tokenizer,
                fl.SetContext(
                    context="text_encoder_pooling", key="end_of_text_index", callback=self.set_end_of_text_index
                ),
                target[1:-2],
                fl.Parallel(
                    fl.Identity(),
                    fl.Chain(
                        target[-2:],
                        projection
                        or fl.Linear(
                            in_features=1280, out_features=1280, bias=False, device=target.device, dtype=target.dtype
                        ),
                        fl.Lambda(func=self.pool),
                    ),
                ),
            )

    def init_context(self) -> Contexts:
        return {"text_encoder_pooling": {"end_of_text_index": []}}

    def __call__(self, text: str) -> tuple[Float[Tensor, "1 77 1280"], Float[Tensor, "1 1280"]]:
        return super().__call__(text)

    @property
    def tokenizer(self) -> CLIPTokenizer:
        return self.ensure_find(CLIPTokenizer)

    def set_end_of_text_index(self, end_of_text_index: list[int], tokens: Tensor) -> None:
        position = (tokens == self.tokenizer.end_of_text_token_id).nonzero(as_tuple=True)[1].item()
        end_of_text_index.append(cast(int, position))

    def pool(self, x: Float[Tensor, "1 77 1280"]) -> Float[Tensor, "1 1280"]:
        end_of_text_index = self.use_context(context_name="text_encoder_pooling").get("end_of_text_index", [])
        assert len(end_of_text_index) == 1, "End of text index not found."
        return x[:, end_of_text_index[0], :]


class DoubleTextEncoder(fl.Chain):
    def __init__(
        self,
        text_encoder_l: CLIPTextEncoderL | None = None,
        text_encoder_g: CLIPTextEncoderG | None = None,
        projection: fl.Linear | None = None,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        text_encoder_l = text_encoder_l or CLIPTextEncoderL(device=device, dtype=dtype)
        text_encoder_g = text_encoder_g or CLIPTextEncoderG(device=device, dtype=dtype)
        super().__init__(
            fl.Parallel(text_encoder_l[:-2], text_encoder_g),
            fl.Lambda(func=self.concatenate_embeddings),
        )
        TextEncoderWithPooling(target=text_encoder_g, projection=projection).inject(parent=self.Parallel)

    def __call__(self, text: str) -> tuple[Float[Tensor, "1 77 2048"], Float[Tensor, "1 1280"]]:
        return super().__call__(text)

    def concatenate_embeddings(
        self, text_embedding_l: Tensor, text_embedding_with_pooling: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        text_embedding_g, pooled_text_embedding = text_embedding_with_pooling
        text_embedding = cat(tensors=[text_embedding_l, text_embedding_g], dim=-1)
        return text_embedding, pooled_text_embedding
