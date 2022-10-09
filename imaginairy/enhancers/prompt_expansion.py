import gzip
import os.path
import random
import re
from functools import lru_cache
from string import Formatter

from imaginairy import PKG_ROOT

DEFAULT_PROMPT_LIBRARY_PATHS = [
    os.path.join(PKG_ROOT, "vendored", "noodle_soup_prompts"),
    os.path.join(PKG_ROOT, "enhancers", "phraselists"),
]
formatter = Formatter()
PROMPT_EXPANSION_PATTERN = re.compile(r"[|a-z0-9_ -]+")


@lru_cache()
def prompt_library_filepaths(prompt_library_paths=None):
    """Return all available category/filepath pairs"""
    prompt_library_paths = [] if not prompt_library_paths else prompt_library_paths
    combined_prompt_library_filepaths = {}
    for prompt_path in DEFAULT_PROMPT_LIBRARY_PATHS + prompt_library_paths:
        library_prompts = prompt_library_filepath(prompt_path)
        combined_prompt_library_filepaths.update(library_prompts)

    return combined_prompt_library_filepaths


@lru_cache()
def category_list(prompt_library_paths=None):
    """Return the names of available phrase-lists"""
    categories = list(prompt_library_filepaths(prompt_library_paths).keys())
    categories.sort()
    return categories


@lru_cache()
def prompt_library_filepath(library_path):
    lookup = {}

    for filename in os.listdir(library_path):
        if "." not in filename:
            continue
        base_filename, ext = filename.split(".", maxsplit=1)
        if ext in {"txt.gz", "txt"}:
            lookup[base_filename.lower()] = os.path.join(library_path, filename)
    return lookup


@lru_cache(maxsize=100)
def get_phrases(category_name, prompt_library_paths=None):
    category_name = category_name.lower()
    lookup = prompt_library_filepaths(prompt_library_paths)
    try:
        filepath = lookup[category_name]
    except KeyError as e:
        raise LookupError(
            f"'{category_name}' is not a valid prompt expansion category. Could not find the txt file."
        ) from e
    _open = open
    if filepath.endswith(".gz"):
        _open = gzip.open

    with _open(filepath, "rb") as f:
        lines = f.readlines()
        phrases = [line.decode("utf-8").strip() for line in lines]
    return phrases


def expand_prompts(prompt_text, n=1, prompt_library_paths=None):
    """
    Replaces {vars} with random samples of corresponding phraselists

    Example:
        p = "a happy {animal}"
        prompts = expand_prompts(p, n=2)
        assert prompts = [
            "a happy dog",
            "a happy cat"
        ]

    """
    prompt_parts = list(formatter.parse(prompt_text))
    field_names = []
    for literal_text, field_name, format_spec, conversion in prompt_parts:  # noqa
        if field_name:
            field_name = field_name.lower()
            if not PROMPT_EXPANSION_PATTERN.match(field_name):
                raise ValueError(
                    "Invalid prompt expansion. Only a-z0-9_|- characters permitted. "
                )
            field_names.append(field_name)

    phrases = []
    for field_name in field_names:
        field_phrases = []
        expansion_tokens = [t.strip() for t in field_name.split("|")]
        for token in expansion_tokens:
            token = token.strip()
            if token.startswith("_") and token.endswith("_"):
                category_name = token.strip("_")
                category_phrases = get_phrases(
                    category_name, prompt_library_paths=prompt_library_paths
                )
                field_phrases.extend(category_phrases)
            else:
                field_phrases.append(token)
        phrases.append(field_phrases)

    for values in get_random_non_repeating_combination(n, *phrases):
        # value_lookup = zip(field_names, values)
        field_count = 0
        output_prompt = ""
        for literal_text, field_name, format_spec, conversion in prompt_parts:

            output_prompt += literal_text
            if field_name:
                output_prompt += values[field_count]
                field_count += 1
        yield output_prompt


def get_random_non_repeating_combination(  # noqa
    n=1, *sequences, allow_oversampling=True
):
    """
    Efficiently return a non-repeating random sample of the product sequences.

    Will repeat if n > num_total_possible combinations and allow_oversampling=True

    Will also potentially repeat after 1_000_000 combinations.
    """
    n_combinations = 1
    for sequence in sequences:
        n_combinations *= len(sequence)

    while n > 0:
        sub_n = n
        if n > n_combinations and allow_oversampling:
            sub_n = n_combinations
        sub_n = min(1_000_000, sub_n)

        indices = random.sample(range(n_combinations), sub_n)

        for idx in indices:
            values = []
            for sequence in sequences:
                seq_idx = idx % len(sequence)
                values.append(sequence[seq_idx])
                idx = idx // len(sequence)
            yield values
        n -= sub_n
