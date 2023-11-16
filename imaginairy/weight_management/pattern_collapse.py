def find_state_dict_key_patterns(patterns):
    """Given a list of state_dict keys, collapse similar keys into patterns.

    For example, if the keys are:

    foo.bar.0.baz
    foo.bar.1.baz

    Then the pattern will be:

    foo.bar.(0|1).baz

    """
    prev_pattern_count = len(patterns) + 1

    # keep running the pattern collapse function until the list of patterns doesn't get any smaller
    while prev_pattern_count > len(patterns):
        prev_pattern_count = len(patterns)
        prev_pattern_count_sub = len(patterns) + 1
        while prev_pattern_count_sub > len(patterns):
            prev_pattern_count_sub = len(patterns)
            patterns = _collapse_patterns(patterns)
        prev_pattern_count_sub = len(patterns) + 1
        while prev_pattern_count_sub > len(patterns):
            prev_pattern_count_sub = len(patterns)
            patterns = _collapse_patterns(patterns, reverse_sort=True)

    return patterns


def prefix_only(k):
    return k.rsplit(".", 1)[0]


def nested_dict_from_keys(keys):
    output = {}
    for key in keys:
        parts = key.split(".")
        # Start from the root of the output and iteratively go deeper
        current_level = output
        for part in parts:
            # If the key doesn't exist at the current level, create a new dict
            if part not in current_level:
                current_level[part] = {}
            # Go one level deeper
            current_level = current_level[part]
    return output


def _collapse_patterns(keys, reverse_sort=False):
    keys = keys.copy()
    keys = [k.split(".") for k in keys]
    if reverse_sort:
        keys.sort(key=lambda k: (len(k), list(reversed(str(k)))))
    else:
        keys.sort(key=lambda k: (len(k), k))
    new_key_patterns = []
    curr_key = None
    for k in keys:
        if curr_key is None:
            curr_key = k
            continue
        single_diff_index = get_single_difference(curr_key, k)
        if single_diff_index is None:
            new_key_patterns.append(curr_key)
            curr_key = k
        else:
            cur_part_val = curr_key[single_diff_index]
            key_part_val = k[single_diff_index]
            if "(" in key_part_val:
                key_vals = key_part_val.strip("()").split("|")
            else:
                key_vals = [key_part_val]
            if "(" in cur_part_val:
                vals = cur_part_val.strip("()").split("|")
            else:
                vals = [cur_part_val]
            vals.extend(key_vals)
            vals.sort()
            try:
                vals = [int(v) for v in vals]
                vals.sort()
                vals = [str(v) for v in vals]
            except ValueError:
                pass
            new_cur_part_val = "(" + "|".join(vals) + ")"
            curr_key[single_diff_index] = new_cur_part_val
    new_key_patterns.append(curr_key)
    new_key_patterns = [".".join(p) for p in new_key_patterns]
    new_key_patterns.sort()
    return new_key_patterns


def get_single_difference(a, b):
    """
    Given two list of strings, if only a single string differs between the two lists, return the index of the differing string.
    """
    if len(a) != len(b):
        return None
    diff_count = 0
    diff_index = None
    for i, (asub, bsub) in enumerate(zip(a, b)):
        if asub != bsub:
            diff_count += 1
            diff_index = i
        if diff_count > 1:
            break

    if diff_count == 1:
        return diff_index
    return None
