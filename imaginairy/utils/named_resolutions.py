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


def get_named_resolution(resolution: str):
    resolution = resolution.upper()

    size = _NAMED_RESOLUTIONS.get(resolution)

    if size is None:
        # is it WIDTHxHEIGHT format?
        try:
            width, height = resolution.split("X")
            size = (int(width), int(height))
        except ValueError:
            pass

    if size is None:
        # is it just a single number?
        with contextlib.suppress(ValueError):
            size = (int(resolution), int(resolution))

    if size is None:
        msg = f"Unknown resolution: {resolution}"
        raise ValueError(msg)

    return size
