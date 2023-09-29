import gzip
import json
import os.path
import time
from contextlib import contextmanager

CURDIR = os.path.dirname(__file__)

excluded_prefixes = ["identity", "gender", "body", "celeb", "color"]
excluded_words = {
    "sex",
    "sexy",
    "sex appeal",
    "sex symbol",
    "young",
    "youth",
    "youthful",
    "child",
    "baby",
}
category_renames = {
    "3d-terms": "3d-term",
    "animals": "animal",
    "camera": "camera-model",
    "camera-manu": "camera-brand",
    "cosmic-terms": "cosmic-term",
    "details": "adj-detailed",
    "foods": "food",
    "games": "video-game",
    "movement": "art-movement",
    "noun-emote": "adj-emotion",
    "natl-park": "national-park",
    "portrait-type": "body-pose",
    "punk": "punk-style",
    "site": "art-site",
    "tree": "tree-species",
    "water": "body-of-water",
    "wh-site": "world-heritage-site",
}


@contextmanager
def timed(description):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    duration = end - start
    print(f"{description} {duration:2f}")


def make_txts():
    src_json = f"{CURDIR}/../downloads/noodle-soup-prompts/nsp_pantry.json"
    dst_folder = f"{CURDIR}/../imaginairy/vendored/noodle_soup_prompts"
    with open(src_json, encoding="utf-8") as f:
        prompts = json.load(f)
    categories = []
    for c in prompts:
        if any(c.startswith(p) for p in excluded_prefixes):
            continue
        categories.append(c)
    categories.sort()
    for c in categories:
        print((c, len(prompts[c])))
        filtered_phrases = [p.lower() for p in prompts[c] if p not in excluded_words]
        renamed_c = category_renames.get(c, c)
        with gzip.open(f"{dst_folder}/{renamed_c}.txt.gz", "wb") as f:
            for p in filtered_phrases:
                f.write(f"{p}\n".encode())


if __name__ == "__main__":
    make_txts()
