from collections import defaultdict


def print_break(length: int):
    print("".join("-" for _ in range(length)))


def nested_dict():
    return defaultdict(nested_dict)


def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        return {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d
