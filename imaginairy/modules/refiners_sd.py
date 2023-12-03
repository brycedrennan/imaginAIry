import math
from typing import Literal

import torch
from refiners.fluxion.layers.attentions import ScaledDotProductAttention
from refiners.fluxion.layers.chain import ChainError
from refiners.foundationals.latent_diffusion import (
    StableDiffusion_1 as RefinerStableDiffusion_1,
    StableDiffusion_1_Inpainting as RefinerStableDiffusion_1_Inpainting,
)
from refiners.foundationals.latent_diffusion.stable_diffusion_1.model import (
    SD1Autoencoder,
)
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

from imaginairy.feather_tile import rebuild_image, tile_image
from imaginairy.modules.autoencoder import logger

TileModeType = Literal["", "x", "y", "xy"]


def _tile_mode_conv2d_conv_forward(
    self, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor  # noqa
):
    if self.padding_modeX == self.padding_modeY:
        self.padding_mode = self.padding_modeX
        return self._orig_conv_forward(input, weight, bias)

    w1 = F.pad(input, self.paddingX, mode=self.padding_modeX)
    del input

    w2 = F.pad(w1, self.paddingY, mode=self.padding_modeY)
    del w1

    return F.conv2d(w2, weight, bias, self.stride, _pair(0), self.dilation, self.groups)


class TileModeMixin(nn.Module):
    def set_tile_mode(self, tile_mode: TileModeType = ""):
        """
        For creating seamless tile images.

        Args:
            tile_mode: One of "", "x", "y", "xy". If "x", the image will be tiled horizontally. If "y", the image will be
                tiled vertically. If "xy", the image will be tiled both horizontally and vertically.
        """

        tile_x = "x" in tile_mode
        tile_y = "y" in tile_mode
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if not hasattr(m, "_orig_conv_forward"):
                    # patch with a function that can handle tiling in a single direction
                    m._initial_padding_mode = m.padding_mode
                    m._orig_conv_forward = m._conv_forward
                    m._conv_forward = _tile_mode_conv2d_conv_forward.__get__(
                        m, nn.Conv2d
                    )
                m.padding_modeX = "circular" if tile_x else "constant"
                m.padding_modeY = "circular" if tile_y else "constant"
                if m.padding_modeY == m.padding_modeX:
                    m.padding_mode = m.padding_modeX
                m.paddingX = (
                    m._reversed_padding_repeated_twice[0],
                    m._reversed_padding_repeated_twice[1],
                    0,
                    0,
                )
                m.paddingY = (
                    0,
                    0,
                    m._reversed_padding_repeated_twice[2],
                    m._reversed_padding_repeated_twice[3],
                )


class StableDiffusion_1(TileModeMixin, RefinerStableDiffusion_1):
    pass


class StableDiffusion_1_Inpainting(TileModeMixin, RefinerStableDiffusion_1_Inpainting):
    def compute_self_attention_guidance(
        self,
        x: Tensor,
        noise: Tensor,
        step: int,
        *,
        clip_text_embedding: Tensor,
        **kwargs: Tensor,
    ) -> Tensor:
        sag = self._find_sag_adapter()
        assert sag is not None
        assert self.mask_latents is not None
        assert self.target_image_latents is not None

        degraded_latents = sag.compute_degraded_latents(
            scheduler=self.scheduler,
            latents=x,
            noise=noise,
            step=step,
            classifier_free_guidance=True,
        )

        negative_embedding, _ = clip_text_embedding.chunk(2)
        timestep = self.scheduler.timesteps[step].unsqueeze(dim=0)
        self.set_unet_context(
            timestep=timestep, clip_text_embedding=negative_embedding, **kwargs
        )
        x = torch.cat(
            tensors=(degraded_latents, self.mask_latents, self.target_image_latents),
            dim=1,
        )
        degraded_noise = self.unet(x)

        return sag.scale * (noise - degraded_noise)


class SD1AutoencoderSliced(SD1Autoencoder):
    max_chunk_size = 2048
    min_chunk_size = 64

    def encode(self, x: Tensor) -> Tensor:
        return self.sliced_encode(x)

    def sliced_encode(self, x: Tensor, chunk_size: int = 128 * 8) -> Tensor:
        """
        Encodes the image in slices (for lower memory usage).
        """
        b, c, h, w = x.size()
        final_tensor = torch.zeros(
            [1, 4, math.ceil(h / 8), math.ceil(w / 8)], device=x.device
        )
        overlap_pct = 0.5

        for x_img in x.split(1):
            chunks = tile_image(
                x_img, tile_size=chunk_size, overlap_percent=overlap_pct
            )
            encoded_chunks = [super(SD1Autoencoder, self).encode(ic) for ic in chunks]

            final_tensor = rebuild_image(
                encoded_chunks,
                base_img=final_tensor,
                tile_size=chunk_size // 8,
                overlap_percent=overlap_pct,
            )

        return final_tensor

    def decode(self, x):
        while self.__class__.max_chunk_size > self.__class__.min_chunk_size:
            if self.max_chunk_size**2 > x.shape[2] * x.shape[3]:
                try:
                    return self.decode_all_at_once(x)
                except ChainError as e:
                    if "OutOfMemoryError" not in str(e):
                        raise
                    self.__class__.max_chunk_size = (
                        int(math.sqrt(x.shape[2] * x.shape[3])) // 2
                    )
                    logger.info(
                        f"Ran out of memory. Trying tiled decode with chunk size {self.__class__.max_chunk_size}"
                    )
            else:
                try:
                    return self.decode_sliced(x, chunk_size=self.max_chunk_size)
                except ChainError as e:
                    if "OutOfMemoryError" not in str(e):
                        raise
                    self.__class__.max_chunk_size = self.max_chunk_size // 2
                    self.__class__.max_chunk_size = max(
                        self.__class__.max_chunk_size, self.__class__.min_chunk_size
                    )
                    logger.info(
                        f"Ran out of memory. Trying tiled decode with chunk size {self.__class__.max_chunk_size}"
                    )
        raise RuntimeError("Could not decode image")

    def decode_all_at_once(self, x: Tensor) -> Tensor:
        decoder = self[1]
        x = decoder(x / self.encoder_scale)
        return x

    def decode_sliced(self, x, chunk_size=128):
        """
        decodes the tensor in slices.

        This results in image portions that don't exactly match, so we overlap, feather, and merge to reduce
        (but not completely eliminate) impact.
        """
        b, c, h, w = x.size()
        final_tensor = torch.zeros([1, 3, h * 8, w * 8], device=x.device)
        for x_latent in x.split(1):
            decoded_chunks = []
            overlap_pct = 0.5
            chunks = tile_image(
                x_latent, tile_size=chunk_size, overlap_percent=overlap_pct
            )

            for latent_chunk in chunks:
                # latent_chunk = self.post_quant_conv(latent_chunk)
                dec = self.decode_all_at_once(latent_chunk)
                decoded_chunks.append(dec)
            final_tensor = rebuild_image(
                decoded_chunks,
                base_img=final_tensor,
                tile_size=chunk_size * 8,
                overlap_percent=overlap_pct,
            )

            return final_tensor


def add_sliced_attention_to_scaled_dot_product_attention(cls):
    """
    Patch refiners ScaledDotProductAttention so that it uses sliced attention

    It reduces peak memory usage.
    """

    def _sliced_attention(self, query, key, value, slice_size, is_causal=None):
        _, num_queries, _ = query.shape
        output = torch.zeros_like(query)
        for start_idx in range(0, num_queries, slice_size):
            end_idx = min(start_idx + slice_size, num_queries)
            output[:, start_idx:end_idx, :] = self._process_attention(
                query[:, start_idx:end_idx, :], key, value, is_causal
            )
        return output

    cls._sliced_attention = _sliced_attention

    def new_forward(self, query, key, value, is_causal=None):
        return self._sliced_attention(
            query, key, value, is_causal=is_causal, slice_size=2048
        )

    cls.forward = new_forward

    def _process_attention(self, query, key, value, is_causal=None):
        return self.merge_multi_head(
            x=self.dot_product(
                query=self.split_to_multi_head(query),
                key=self.split_to_multi_head(key),
                value=self.split_to_multi_head(value),
                is_causal=(
                    is_causal
                    if is_causal is not None
                    else (self.is_causal if self.is_causal is not None else False)
                ),
            )
        )

    cls._process_attention = _process_attention
    logger.debug(f"Patched {cls.__name__} with sliced attention")


add_sliced_attention_to_scaled_dot_product_attention(ScaledDotProductAttention)
