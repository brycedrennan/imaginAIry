from torch.nn import PixelUnshuffle as _PixelUnshuffle

from imaginairy.vendored.refiners.fluxion.layers.module import Module


class PixelUnshuffle(_PixelUnshuffle, Module):
    def __init__(self, downscale_factor: int):
        _PixelUnshuffle.__init__(self, downscale_factor=downscale_factor)
