import torch

from imaginairy.vendored.refiners.foundationals.dinov2.vit import ViT

# TODO: add preprocessing logic like
# https://github.com/facebookresearch/dinov2/blob/2302b6b/dinov2/data/transforms.py#L77


class DINOv2_small(ViT):
    def __init__(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            embedding_dim=384,
            patch_size=14,
            image_size=518,
            num_layers=12,
            num_heads=6,
            device=device,
            dtype=dtype,
        )


class DINOv2_base(ViT):
    def __init__(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            embedding_dim=768,
            patch_size=14,
            image_size=518,
            num_layers=12,
            num_heads=12,
            device=device,
            dtype=dtype,
        )


class DINOv2_large(ViT):
    def __init__(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            embedding_dim=1024,
            patch_size=14,
            image_size=518,
            num_layers=24,
            num_heads=16,
            device=device,
            dtype=dtype,
        )


# TODO: implement SwiGLU layer
# class DINOv2_giant2(ViT):
#     def __init__(
#         self,
#         device: torch.device | str | None = None,
#         dtype: torch.dtype | None = None,
#     ) -> None:
#         super().__init__(
#             embedding_dim=1536,
#             patch_size=14,
#             image_size=518,
#             num_layers=40,
#             num_heads=24,
#             device=device,
#             dtype=dtype,
#         )


class DINOv2_small_reg(ViT):
    def __init__(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            embedding_dim=384,
            patch_size=14,
            image_size=518,
            num_layers=12,
            num_heads=6,
            num_registers=4,
            device=device,
            dtype=dtype,
        )


class DINOv2_base_reg(ViT):
    def __init__(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            embedding_dim=768,
            patch_size=14,
            image_size=518,
            num_layers=12,
            num_heads=12,
            num_registers=4,
            device=device,
            dtype=dtype,
        )


class DINOv2_large_reg(ViT):
    def __init__(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            embedding_dim=1024,
            patch_size=14,
            image_size=518,
            num_layers=24,
            num_heads=16,
            num_registers=4,
            device=device,
            dtype=dtype,
        )


# TODO: implement SwiGLU layer
# class DINOv2_giant2_reg(ViT):
#     def __init__(
#         self,
#         device: torch.device | str | None = None,
#         dtype: torch.dtype | None = None,
#     ) -> None:
#         super().__init__(
#             embedding_dim=1536,
#             patch_size=14,
#             image_size=518,
#             num_layers=40,
#             num_heads=24,
#             num_registers=4,
#             device=device,
#             dtype=dtype,
#         )
