from collections import defaultdict
from thefuzz import fuzz

def print_break(length: int):
    print("".join("-" for _ in range(length)))


def nested_dict():
    return defaultdict(nested_dict)


def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        return {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d

def auto_correct(orientation_name: str) -> str:
    expected_orientations = ["Sag", "Ax", "Cor"]
    best_match = sorted(expected_orientations, key=lambda x: fuzz.ratio(x, orientation_name), reverse=True)[0]
    return best_match
