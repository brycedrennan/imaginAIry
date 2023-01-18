from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("imaginairy")
except PackageNotFoundError:
    __version__ = None
