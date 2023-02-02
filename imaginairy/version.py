def get_version():
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("imaginairy")
    except PackageNotFoundError:
        return None
