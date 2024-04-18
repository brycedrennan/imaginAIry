import logging
import os
import re
import urllib.parse
from functools import lru_cache, wraps

import requests
from huggingface_hub import (
    HfFileSystem,
    HfFolder,
    hf_hub_download as _hf_hub_download,
    try_to_load_from_cache,
)

logger = logging.getLogger(__name__)


def resolve_path_or_url(path_or_url: str, category=None) -> str:
    """
    Resolves a path or url to a local absolute file path

    If the path_or_url is a url, it will be downloaded to the cache directory and the path to the downloaded file will be returned.
    """
    if path_or_url.startswith(("https://", "http://")):
        return get_cached_url_path(url=path_or_url, category=category)
    return os.path.abspath(path_or_url)


def get_cached_url_path(url: str, category=None) -> str:
    """
    Gets the contents of a url, but caches the response indefinitely.

    While we attempt to use the cached_path from huggingface transformers, we fall back
    to our own implementation if the url does not provide an etag header, which `cached_path`
    requires.  We also skip the `head` call that `cached_path` makes on every call if the file
    is already cached.
    """
    if url.startswith("https://huggingface.co"):
        try:
            return huggingface_cached_path(url)
        except (OSError, ValueError):
            pass
    filename = url.split("/")[-1]
    dest = get_cache_dir()
    if category:
        dest = os.path.join(dest, category)
    os.makedirs(dest, exist_ok=True)

    # Replace possibly illegal destination path characters
    safe_filename = re.sub('[*<>:"|?]', "_", filename)
    dest_path = os.path.join(dest, safe_filename)
    if os.path.exists(dest_path):
        return dest_path

    # check if it's saved at previous path and rename it
    old_dest_path = os.path.join(dest, filename)
    if os.path.exists(old_dest_path):
        os.rename(old_dest_path, dest_path)
        return dest_path

    r = requests.get(url)

    with open(dest_path, "wb") as f:
        f.write(r.content)
    return dest_path


def check_huggingface_url_authorized(url: str) -> None:
    if not url.startswith("https://huggingface.co/"):
        return None
    token = HfFolder.get_token()
    headers = {}
    if token is not None:
        headers["authorization"] = f"Bearer {token}"
    response = requests.head(url, allow_redirects=True, headers=headers, timeout=5)
    if response.status_code == 401:
        msg = "Unauthorized access to HuggingFace model. This model requires a huggingface token.  Please login to HuggingFace or set HUGGING_FACE_HUB_TOKEN to your User Access Token. See https://huggingface.co/docs/huggingface_hub/quick-start#login for more information"
        raise HuggingFaceAuthorizationError(msg)
    return None


@wraps(_hf_hub_download)
def hf_hub_download(*args, **kwargs):
    """
    backwards compatible wrapper for huggingface's hf_hub_download.

    they changed the argument name from `use_auth_token` to `token`
    """

    try:
        return _hf_hub_download(*args, **kwargs)
    except TypeError as e:
        if "unexpected keyword argument 'token'" in str(e):
            kwargs["use_auth_token"] = kwargs.pop("token")
            return _hf_hub_download(*args, **kwargs)
        raise


def huggingface_cached_path(url: str) -> str:
    # bypass all the HEAD calls done by the default `cached_path`
    repo, commit_hash, filepath = extract_huggingface_repo_commit_file_from_url(url)
    dest_path = try_to_load_from_cache(
        repo_id=repo, revision=commit_hash, filename=filepath
    )
    from huggingface_hub.file_download import _CACHED_NO_EXIST

    if not dest_path or dest_path == _CACHED_NO_EXIST:
        check_huggingface_url_authorized(url)
        token = HfFolder.get_token()
        logger.info(f"Downloading {url} from huggingface")
        dest_path = hf_hub_download(
            repo_id=repo, revision=commit_hash, filename=filepath, token=token
        )
        # make a refs folder so caching works
        # work-around for
        # https://github.com/huggingface/huggingface_hub/pull/1306
        # https://github.com/brycedrennan/imaginAIry/issues/171
        refs_url = dest_path[: dest_path.index("/snapshots/")] + "/refs/"
        os.makedirs(refs_url, exist_ok=True)
    return dest_path


def extract_huggingface_repo_commit_file_from_url(url):
    parsed_url = urllib.parse.urlparse(url)
    path_components = parsed_url.path.strip("/").split("/")

    repo = "/".join(path_components[0:2])
    assert path_components[2] == "resolve"
    commit_hash = path_components[3]
    filepath = "/".join(path_components[4:])

    return repo, commit_hash, filepath


def download_huggingface_weights(
    base_url: str, sub: str, filename=None, prefer_fp16=True
) -> str:
    """
    Downloads weights from huggingface and returns the path to the downloaded file

    Given a huggingface repo url, folder, and optional filename, download the weights to the cache directory and return the path
    """
    if filename is None:
        # select which weights to download. prefer fp16 safetensors
        data = parse_diffusers_repo_url(base_url)
        fs = HfFileSystem()
        filepaths = fs.ls(
            f"{data['author']}/{data['repo']}/{sub}", revision=data["ref"], detail=False
        )
        filepath = choose_huggingface_weights(filepaths, prefer_fp16=prefer_fp16)
        if not filepath:
            msg = f"Could not find any weights in {base_url}/{sub}"
            raise ValueError(msg)
        filename = filepath.split("/")[-1]
    url = f"{base_url}{sub}/{filename}".replace("/tree/", "/resolve/")
    new_path = get_cached_url_path(url, category="weights")
    return new_path


def choose_huggingface_weights(filenames: list[str], prefer_fp16=True) -> str | None:
    """
    Chooses the best weights file from a list of filenames

    Prefers safetensors format and fp16 dtype
    """
    extension_priority = (".safetensors", ".bin", ".pth", ".pt")
    # filter out any files that don't have a valid extension
    filenames = [f for f in filenames if any(f.endswith(e) for e in extension_priority)]
    filenames_and_extension = [(f, os.path.splitext(f)[1]) for f in filenames]
    # sort by priority
    if prefer_fp16:
        filenames_and_extension.sort(
            key=lambda x: ("fp16" not in x[0], extension_priority.index(x[1]))
        )
    else:
        filenames_and_extension.sort(
            key=lambda x: ("fp16" in x[0], extension_priority.index(x[1]))
        )
    if filenames_and_extension:
        return filenames_and_extension[0][0]
    return None


@lru_cache
def get_cache_dir() -> str:
    xdg_cache_home = os.getenv("XDG_CACHE_HOME", None)
    if xdg_cache_home is None:
        user_home = os.getenv("HOME", None)
        if user_home:
            xdg_cache_home = os.path.join(user_home, ".cache")

    if xdg_cache_home is not None:
        return os.path.join(xdg_cache_home, "imaginairy")

    return os.path.join(os.path.dirname(__file__), ".cached-aimg")


class HuggingFaceAuthorizationError(RuntimeError):
    pass


hf_repo_url_pattern = re.compile(
    r"https://huggingface\.co/(?P<author>[^/]+)/(?P<repo>[^/]+)(/tree/(?P<ref>[a-z0-9]+))?/?$"
)


def parse_diffusers_repo_url(url: str) -> dict[str, str]:
    match = hf_repo_url_pattern.match(url)
    return match.groupdict() if match else {}


def is_diffusers_repo_url(url: str) -> bool:
    result = bool(parse_diffusers_repo_url(url))
    logger.debug(f"{url} is diffusers repo url: {result}")
    return result


def normalize_diffusers_repo_url(url: str) -> str:
    data = parse_diffusers_repo_url(url)
    ref = data["ref"] or "main"
    normalized_url = (
        f"https://huggingface.co/{data['author']}/{data['repo']}/tree/{ref}/"
    )
    return normalized_url
