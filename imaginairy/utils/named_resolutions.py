import contextlib

_NAMED_RESOLUTIONS = {
    "HD": (1280, 720),
    "FHD": (1920, 1080),
    "2K": (2048, 1080),
    "4K": (3840, 2160),
    "UHD": (3840, 2160),
    "8K": (7680, 4320),
    "360p": (640, 360),
    "VGA": (640, 480),
    "SVGA": (800, 600),
    "XGA": (1024, 768),
    "SXGA": (1280, 1024),
    "WXGA+": (1440, 900),
    "HD+": (1600, 900),
    "UXGA": (1600, 1200),
    "WSXGA+": (1680, 1050),
    "WUXGA": (1920, 1200),
    "QWXGA": (2048, 1152),
    "QXGA": (2048, 1536),
    "UWFHD": (2560, 1080),
    "QHD": (2560, 1440),
    "WQXGA": (2560, 1600),
    "UWQHD": (3440, 1440),
    "240p": (426, 240),
    "480p": (854, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "2160p": (3840, 2160),
    "NTSC": (720, 480),
    "PAL": (720, 576),
    "QVGA": (320, 240),
    "WVGA": (800, 480),
    "FWVGA": (854, 480),
    "WSVGA": (1024, 600),
    "HDV": (1440, 1080),
    "WQHD": (2560, 1440),
    "UW-UXGA": (2560, 1080),
    "UHD+": (5120, 2880),
    "8K UHD": (7680, 4320),
    "SVD": (1024, 576),  # stable video diffusion
}

_NAMED_RESOLUTIONS = {k.upper(): v for k, v in _NAMED_RESOLUTIONS.items()}


def normalize_image_size(resolution: str | int | tuple[int, int]) -> tuple[int, int]:
    match resolution:
        case (int(), int()):
            size = resolution
        case int():
            size = resolution, resolution
        case str():
            resolution = resolution.strip().upper()
            resolution = resolution.replace(" ", "").replace("X", ",").replace("*", ",")
            size = _NAMED_RESOLUTIONS.get(resolution.upper())
            if size is None:
                # is it WIDTH,HEIGHT format?
                try:
                    width, height = resolution.split(",")
                    size = int(width), int(height)
                except ValueError:
                    pass
            if size is None:
                # is it just a single number?
                with contextlib.suppress(ValueError):
                    size = (int(resolution), int(resolution))
            if size is None:
                msg = f"Invalid resolution: '{resolution}'"
                raise ValueError(msg)
        case _:
            msg = f"Invalid resolution: {resolution!r}"
            raise ValueError(msg)
    if size[0] <= 0 or size[1] <= 0:
        msg = f"Invalid resolution: {resolution!r}"
        raise ValueError(msg)
    return size
