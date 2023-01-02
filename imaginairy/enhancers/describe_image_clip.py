from functools import lru_cache
from typing import Sequence

import torch
from PIL import Image
from torch import nn

from imaginairy.vendored import clip

device = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache
def get_model():
    model_name = "ViT-L/14"
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess


def find_img_text_similarity(image: Image.Image, phrases: Sequence):
    """Find the likelihood of a list of textual concepts existing in the image."""

    model, preprocess = get_model()
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)

    return find_embed_text_similarity(image_features, phrases)


def find_embed_text_similarity(embed_features, phrases):
    model, _ = get_model()
    text = clip.tokenize(phrases).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)

    probs = cosine_distance(text_features, embed_features)
    probs = [float(p) for p in probs.squeeze(dim=0)]
    phrase_probs = list(zip(phrases, probs))
    phrase_probs.sort(key=lambda r: r[1], reverse=True)

    return phrase_probs


def rank(image_features, text_features, top_count=100):
    similarity = torch.zeros((1, text_features.shape[0])).to(device)
    for i in range(image_features.shape[0]):
        similarity += (
            100.0 * image_features[i].unsqueeze(0) @ text_features.T
        ).softmax(dim=-1)
    similarity /= image_features.shape[0]

    top_probs, top_labels = similarity.cpu().topk(top_count, dim=-1)
    phrase_scores = [
        (top_labels[0][i].numpy(), (top_probs[0][i].numpy() * 100))
        for i in range(top_count)
    ]
    phrase_scores = [(p, s) for p, s in phrase_scores if s > 0.0001]
    phrase_scores.sort(key=lambda ps: ps[1], reverse=True)
    return phrase_scores


def cosine_distance(embeds_a, embeds_b):
    embeds_a = nn.functional.normalize(embeds_a)
    embeds_b = nn.functional.normalize(embeds_b)
    return torch.mm(embeds_a, embeds_b.t())
