import hashlib

from imaginairy.modules.clip_embedders import FrozenCLIPEmbedder
from imaginairy.utils import get_device


def hash_tensor(t):
    t = t.cpu().detach().numpy().tobytes()
    return hashlib.md5(t).hexdigest()


def test_text_conditioning():
    embedder = FrozenCLIPEmbedder()
    embedder.to(get_device())
    neutral_embedding = embedder.encode([""])
    hashed = hash_tensor(neutral_embedding)
    if "mps" in get_device():
        assert hashed == "263e5ee7d2be087d816e094b80ffc546"
    elif "cuda" in get_device():
        assert hashed == "3d7867d5b2ebf15102a9ca9476d63ebc"
