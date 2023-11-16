import itertools
import json
import os
from collections import defaultdict

from imaginairy.weight_management.utils import WEIGHT_INFO_PATH, WEIGHT_MAPS_PATH


def generate_conversion_maps():
    execution_orders_map = defaultdict(dict)
    for filename in os.listdir(WEIGHT_INFO_PATH):
        if not filename.endswith("prefix-execution-order.json"):
            continue

        base_name = filename.split(".", 1)[0]
        model_name, component_name, format_name = base_name.split("_")
        execution_orders_map[(model_name, component_name)][format_name] = filename

    for (model_name, component_name), format_lookup in execution_orders_map.items():
        if len(format_lookup) <= 1:
            continue

        formats = list(format_lookup.keys())
        for format_a, format_b in itertools.permutations(formats, 2):
            filename_a = format_lookup[format_a]
            filename_b = format_lookup[format_b]
            with open(os.path.join(WEIGHT_INFO_PATH, filename_a)) as f:
                execution_order_a = json.load(f)
            with open(os.path.join(WEIGHT_INFO_PATH, filename_b)) as f:
                execution_order_b = json.load(f)

            mapping_filename = (
                f"{model_name}_{component_name}_{format_a}_TO_{format_b}.json"
            )
            mapping_filepath = os.path.join(WEIGHT_MAPS_PATH, mapping_filename)
            print(f"Creating {mapping_filename}...")
            if os.path.exists(mapping_filepath):
                continue

            if len(execution_order_a) != len(execution_order_b):
                print(
                    f"Could not create {mapping_filename} - Execution orders for {format_a} and {format_b} have different lengths"
                )
                continue

            mapping = dict(zip(execution_order_a, execution_order_b))
            mapping_info = {
                "mapping": mapping,
                "source_aliases": {},
                "ignorable_prefixes": [],
            }
            with open(mapping_filepath, "w") as f:
                json.dump(mapping_info, f, indent=2)


if __name__ == "__main__":
    generate_conversion_maps()
