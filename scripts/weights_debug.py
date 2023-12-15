from imaginairy.utils.model_manager import load_tensors


def dotstrings_to_nested_dictionaries(list_of_dotstrings):
    """given a list of dotstrings, return a nested dictionary."""
    nested_dict = {}
    for dotstring in list_of_dotstrings:
        keys = dotstring.split(".")
        d = nested_dict
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = None
    return nested_dict


def display_nested_dictionary(nested_dictionary, num_levels_deep=2):
    """given a nested dictionary, print it out."""

    def _display_nested_dictionary(d, level=0):
        if level >= num_levels_deep:
            return
        for k, v in d.items():
            print("  " * level + k)
            if isinstance(v, dict):
                _display_nested_dictionary(v, level + 1)

    _display_nested_dictionary(nested_dictionary)


def display_weights_structure(weights_path, num_levels_deep=2):
    """given a weights path, display the structure of the weights."""
    print(
        f"Displaying weights structure for {weights_path} to {num_levels_deep} levels deep"
    )
    data = load_tensors(weights_path)
    display_nested_dictionary(
        dotstrings_to_nested_dictionaries(data.keys()), num_levels_deep=num_levels_deep
    )
