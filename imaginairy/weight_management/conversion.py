import os.path
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from imaginairy.weight_management import utils

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class WeightMap:
    model_name: str
    component_name: str
    source_format: str
    dest_format: str

    def __post_init__(self):
        self.model_name = self.model_name.replace("_", "-")
        self.component_name = self.component_name.replace("_", "-")
        self.source_format = self.source_format.replace("_", "-")
        self.dest_format = self.dest_format.replace("_", "-")

        self._loaded_mapping_info = None

    @property
    def filename(self):
        return f"{self.model_name}_{self.component_name}_{self.source_format}_TO_{self.dest_format}.json"

    @property
    def filepath(self):
        return os.path.join(utils.WEIGHT_MAPS_PATH, self.filename)

    @property
    def _mapping_info(self):
        if self._loaded_mapping_info is None:
            import json

            with open(self.filepath) as f:
                self._loaded_mapping_info = json.load(f)
        return self._loaded_mapping_info

    @property
    def mapping(self):
        return self._mapping_info["mapping"]

    @property
    def source_aliases(self):
        return self._mapping_info.get("source_aliases", {})

    @property
    def ignorable_prefixes(self):
        return self._mapping_info.get("ignorable_prefixes", [])

    @property
    def reshapes(self):
        return self._mapping_info.get("reshapes", {})

    @property
    def all_valid_prefixes(self):
        return (
            set(self.mapping.keys())
            | set(self.source_aliases.keys())
            | set(self.ignorable_prefixes)
        )

    def could_convert(self, source_weights):
        source_keys = set(source_weights.keys())
        return source_keys.issubset(self.all_valid_prefixes)

    def cast_weights(self, source_weights):
        converted_state_dict: dict[str, Tensor] = {}
        for source_key in source_weights:
            source_prefix, suffix = source_key.rsplit(sep=".", maxsplit=1)
            # handle aliases
            source_prefix = self.source_aliases.get(source_prefix, source_prefix)
            try:
                target_prefix = self.mapping[source_prefix]
            except KeyError:
                continue
            target_key = ".".join([target_prefix, suffix])
            converted_state_dict[target_key] = source_weights[source_key]

        for key, new_shape in self.reshapes.items():
            converted_state_dict[key] = converted_state_dict[key].reshape(new_shape)

        return converted_state_dict


@lru_cache(maxsize=None)
def load_state_dict_conversion_maps():
    import json

    conversion_maps = {}
    from importlib.resources import files

    for file in files("imaginairy").joinpath("weight_conversion/maps").iterdir():
        if file.is_file() and file.suffix == ".json":
            conversion_maps[file.name] = json.loads(file.read_text())
    return conversion_maps


def cast_weights(
    source_weights, source_model_name, source_component_name, source_format, dest_format
):
    weight_map = WeightMap(
        model_name=source_model_name,
        component_name=source_component_name,
        source_format=source_format,
        dest_format=dest_format,
    )
    return weight_map.cast_weights(source_weights)
