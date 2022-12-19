import hashlib

import pytest

from imaginairy.modules.clip_embedders import FrozenCLIPEmbedder
from imaginairy.utils import get_device


def hash_tensor(t):
    t = t.cpu().detach().numpy().tobytes()
    return hashlib.md5(t).hexdigest()


@pytest.mark.skipif(get_device() == "cpu", reason="Too slow to run on CPU")
def test_text_conditioning():
    embedder = FrozenCLIPEmbedder()
    embedder.to(get_device())
    neutral_embedding = embedder.encode([""])
    hashed = hash_tensor(neutral_embedding)
    assert hashed in {
        "263e5ee7d2be087d816e094b80ffc546",  # mps
        "41818051d7c469fc57d0a940c9d24d82",
        "b5f29fb26bceb60dcde19ec7ec5a0711",
        "88245bdb2a83b49092407fc5b4c473ab",  # ubuntu, torch 1.12.1 cu116
    }
