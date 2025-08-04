"""
utils.py

This script stores miscellaneous utility functions for use throughout the backend.
Note that hazen itself has a utility script too, so this script is designed for more top-level backend utility.

Written by Nathan Crossley 2025

"""

from pathlib import Path
from collections import defaultdict
from thefuzz import fuzz
import pydicom


def nested_dict() -> defaultdict:
    """Returns a nested defaultdict that automatically creates deeper levels
    of dictionaries as needed. Useful for building recursive or multi-level
    dictionary structures without manually checking or initializing intermediate keys.
    """
    return defaultdict(nested_dict)


def defaultdict_to_dict(d: defaultdict) -> dict:
    """Recursively converts a nested defaultdict (like nested_dict) back
    to a regular dict.

    Args:
        d (defaultdict): defaultdict to convert to a regular dict.

    Returns:
        dict: Regular dict with the same multi-level structure as the input defaultdict.
    """
    # recursively converts a nested default_dict back for regular dict
    if isinstance(d, defaultdict):
        return {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d


def chained_get(main_dict: dict, *keys, default: any = "N/A") -> any:
    """Safe getter for nested results dict.
    Iterates through keys and returns the value at the end of the chain if it exists.
    Otherwise, returns the default value.

    Args:
        main_dict (dict): Dictionary to search in.
        default (any, optional): Default return value if key chain fails.

    Returns:
        any: Value at the end of the key chain or default value.
    """
    d = main_dict.copy()
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, None)
            if d is None:
                return default
    return d


def substring_matcher(string: str, strings_to_search: list[str]) -> str:
    """Finds the best match for a string from a list of strings using fuzzy matching.

    Args:
        string (str): String to match.
        strings_to_search (list[str]): Strings to check similarity against main string.

    Returns:
        str: Best matching string from the list.
    """

    # order list of test strings based on string similarity - to get most similar string
    best_match = sorted(
        strings_to_search,
        key=lambda x: fuzz.partial_ratio(x.lower(), string.lower()),
        reverse=True,
    )[0]
    return best_match


def quick_check_dicom(file: Path) -> bool:
    """Quickly checks if a file is a DICOM file by looking for
    the DICM string in the first 128 bytes.

    Args:
        file (Path): Path of file to check nature of.

    Returns:
        bool: True if file is a DICOM file, False otherwise.
    """
    # Try to check if file is DICOM
    try:
        # opens file
        with file.open("rb") as f:

            # skips first 128 bytes
            f.seek(128)

            # reads next 4 bytes - return true if DICOM in nature
            if f.read(4) == b"DICM":
                return True

    # Fail silently so can employ backup DICOM check
    except:
        pass

    # Check if file is DICOM using more robust, slower pydicom method
    try:
        pydicom.dcmread(file, stop_before_pixels=True)
        return True

    # If file not DICOM, return False
    except:
        return False
