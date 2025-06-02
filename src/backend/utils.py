from collections import defaultdict
from thefuzz import fuzz
from pathlib import Path
import pydicom


def nested_dict():
    """Creates a nested defaultdict with a default factory of another defaultdict."""
    return defaultdict(nested_dict)


def defaultdict_to_dict(d: defaultdict) -> dict:
    """Recursively converts a defaultdict to a regular dict.

    Args:
        d (defaultdict): defaultdict to convert to a regular dict.

    Returns:
        dict: Regular dict with the same structure as the input defaultdict.
    """
    if isinstance(d, defaultdict):
        return {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d


def substring_matcher(string: str, strings_to_search: list[str]) -> str:
    """Finds the best match for a string from a list of strings using fuzzy matching.

    Args:
        string (str): String to match.
        strings_to_search (list[str]): Strings to check similarity against main string.

    Returns:
        str: Best matching string from the list.
    """
    best_match = sorted(
        strings_to_search,
        key=lambda x: fuzz.ratio(x.lower(), string.lower()),
        reverse=True,
    )[0]
    return best_match


def quick_check_dicom(file: Path) -> bool:
    """Quickly checks if a file is a DICOM file by looking for
    the DICM string in the first 128 bytes.

    Args:
        file (Path): Path to file to check.

    Returns:
        bool: True if file is a DICOM file, False otherwise.
    """
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
