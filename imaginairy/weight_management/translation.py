import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Dict

import torch
from refiners.fluxion import load_from_safetensors
from torch import device as Device

logger = logging.getLogger(__name__)


TensorDict = Dict[str, torch.Tensor]


@dataclass
class WeightTranslationMap:
    name_map: dict[str, str | None] = field(default_factory=dict)
    regex_map: dict[str, str | None] = field(default_factory=dict)
    ignore_prefixes: list[str] = field(default_factory=list)
    source_aliases: dict[str, str] = field(default_factory=dict)
    reshapes: dict[str, tuple[int, ...]] = field(default_factory=dict)

    def load_and_translate_weights(
        self, source_path: str, device: Device | str = "cpu"
    ) -> TensorDict:
        extension = source_path.split(".")[-1]
        if extension in ["pth", "pt", "bin"]:
            source_weights = torch.load(source_path, map_location="cpu")

        elif extension in ["safetensors"]:
            source_weights = load_from_safetensors(source_path, device=device)
        else:
            msg = f"Unsupported extension {extension}"
            raise ValueError(msg)

        return self.translate_weights(source_weights)

    def translate_weights(self, source_weights: TensorDict) -> TensorDict:
        return translate_weights(source_weights=source_weights, weight_map=self)

    def add_regex_replacer(self, find_pattern: str, replace_pattern: str):
        orig_size = len(self.name_map)
        # see which keys of name_map
        find_pattern_c = re.compile(find_pattern)
        matched_keys = [k for k in self.name_map if find_pattern_c.match(k)]
        # print(f"Pattern {find_pattern} matched {len(matched_keys)} keys")
        for k in matched_keys:
            match = find_pattern_c.match(k)
            if not match:
                continue
            data = match.groupdict()
            new_k = render_fstring(replace_pattern, data)
            expected_k = self.name_map[k]
            assert new_k == expected_k
            # print(f"Replacing {k} with {new_k}")
            del self.name_map[k]

        self.regex_map[find_pattern] = replace_pattern
        print(
            f"Adding pattern reduced name_map from {orig_size} to {len(self.name_map)}"
        )

    def save(self, path):
        import json
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load(cls, path):
        import json

        with open(path) as f:
            d = json.load(f)
        return cls(**d)


def translate_weights(
    source_weights: TensorDict, weight_map: WeightTranslationMap
) -> TensorDict:
    new_state_dict: TensorDict = {}
    # print(f"Translating {len(source_weights)} weights")
    # print(f"Using {len(weight_map.name_map)} name mappings")
    # print(source_weights.keys())

    source_weights = flatten_dict(source_weights)

    for source_key in list(source_weights.keys()):
        source_key = weight_map.source_aliases.get(source_key, source_key)
        try:
            target_key = weight_map.name_map[source_key]
            # print(f"Found {source_prefix} -> {target_prefix}")
        except KeyError:
            continue
        if target_key is None:
            # mapped to None means we ignore it
            source_weights.pop(source_key)
        else:
            # print(f"Adding {target_key}")
            new_state_dict[target_key] = source_weights.pop(source_key)

    for source_key in list(source_weights.keys()):
        try:
            source_prefix, suffix = source_key.rsplit(sep=".", maxsplit=1)
        except ValueError:
            # no dots
            continue
        # print(f"Checking {source_prefix} {suffix}")

        source_prefix = weight_map.source_aliases.get(source_prefix, source_prefix)
        try:
            target_prefix = weight_map.name_map[source_prefix]
            # print(f"Found {source_prefix} -> {target_prefix}")
        except KeyError:
            continue
        if target_prefix is None:
            # mapped to None means we ignore it
            source_weights.pop(source_key)
            continue
        else:
            target_key = ".".join([target_prefix, suffix])
            # print(f"Adding {target_key}")
            new_state_dict[target_key] = source_weights.pop(source_key)

    for source_key in list(source_weights.keys()):
        try:
            source_prefix, suffix = source_key.rsplit(sep=".", maxsplit=1)
        except ValueError:
            # no dots
            continue
        for pattern, replace_pattern in weight_map.regex_map.items():
            match = re.match(pattern, source_prefix)
            if match:
                match_data = match.groupdict()
                new_k = render_fstring(replace_pattern, match_data)
                new_k = ".".join([new_k, suffix])
                new_state_dict[new_k] = source_weights.pop(source_key)

    if source_weights:
        msg = f"Unmapped keys: {list(source_weights.keys())}"
        print(msg)
        for k in source_weights:
            if isinstance(source_weights[k], torch.Tensor):
                print(f"  {k}: {source_weights[k].shape}")
            else:
                print(f"  {k}: {repr(source_weights[k])[:100]}")

    if weight_map.reshapes:
        for key, new_shape in weight_map.reshapes.items():
            if key in new_state_dict:
                new_state_dict[key] = new_state_dict[key].reshape(new_shape)

    return new_state_dict


def flatten_dict(d, sep="."):
    """
    Flattens a nested dictionary into a dictionary with dot-separated keys.
    This function removes items from the original dictionary as they are added to the new one.
    The function uses an iterative approach instead of recursion.

    Parameters:
    d (dict): The dictionary to flatten.
    sep (str): The separator to use between keys.

    Returns:
    dict: A flattened dictionary.
    """
    flat_dict = {}
    stack = [("", d)]

    while stack:
        parent_key, current_dict = stack.pop()
        keys = list(current_dict.keys())  # Create a list of keys to avoid RuntimeError
        for k in keys:
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(current_dict[k], dict):
                stack.append((new_key, current_dict[k]))
            else:
                flat_dict[new_key] = current_dict.pop(k)

    return flat_dict


def render_fstring(fstring, variables):
    """
    Render a string formatted like an f-string using the provided variables.

    DANGER: This is a security risk if the fstring is user-provided.

    Args:
    fstring (str): The template string with placeholders for variables.
    variables (dict): A dictionary containing the variables to be used in the f-string.

    Returns:
    str: The rendered string with variables substituted.
    """
    # Use locals().update to add the variables to the local scope
    locals().update(variables)

    # Evaluate the f-string using eval with an f-string formatted string
    return eval(f'f"""{fstring}"""')
