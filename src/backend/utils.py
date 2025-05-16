from collections import defaultdict
from thefuzz import fuzz
from pathlib import Path
import pydicom


def nested_dict():
    return defaultdict(nested_dict)


def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        return {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d


def substring_matcher(string: str, strings_to_search: list[str]) -> str:
    best_match = sorted(
        strings_to_search,
        key=lambda x: fuzz.ratio(x.lower(), string.lower()),
        reverse=True,
    )[0]
    return best_match


def quick_check_dicom(file: Path) -> bool:
    try:
        with file.open("rb") as f:
            f.seek(128)
            if f.read(4) == b"DICM":
                return True
    except:
        pass

    try:
        pydicom.dcmread(file, stop_before_pixels=True)
        return True
    except:
        return False
