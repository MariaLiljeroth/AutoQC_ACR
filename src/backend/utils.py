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
import csv
from collections import defaultdict
from typing import Any, List
import numpy as np


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

# utils.py


def dump_nested_dict_to_csv(d: dict, out_subdir: Path, filename: str = "results_dump.csv"):
    """
    Dumps a nested dictionary to a CSV file.
    Nested keys are flattened into columns; lists or arrays are expanded.
    The CSV is saved one level up from out_subdir.
    """
    def flatten(prefix: List[str], value: Any, rows: List[List[Any]]):
        if isinstance(value, dict):
            for k, v in value.items():
                flatten(prefix + [k], v, rows)
        elif isinstance(value, list):
            rows.append(prefix + value)
        else:
            rows.append(prefix + [value])

    def normalize_cell(value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, (float, np.floating)):
            return "" if np.isnan(value) else value
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, str):
            return "" if value.strip().lower() in {"", "nan", "n/a", "none"} else value
        return value

    rows: List[List[Any]] = []
    flatten([], d, rows)
    rows = [[normalize_cell(cell) for cell in row] for row in rows]

    max_depth = max(len(row) for row in rows)
    header = [f"Level_{i+1}" for i in range(max_depth - 1)] + ["Value"]
    padded_rows = [row + [""] * (max_depth - len(row)) for row in rows]

    parent_dir = Path(out_subdir).parent
    csv_path = parent_dir / filename

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(padded_rows)

    print(f"Nested dictionary successfully dumped to CSV at: {csv_path}")


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


def coerce_numeric(value: Any, default: Any = np.nan) -> Any:
    """Cast common scalar inputs to float while preserving missing values as NaN."""
    if value is None:
        return default

    if isinstance(value, dict):
        for key in ("measured", "value", "result"):
            if key in value:
                return coerce_numeric(value[key], default)
        return default

    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return coerce_numeric(value[0], default)
        return default

    if isinstance(value, (str, bytes)):
        cleaned = str(value).strip()
        if cleaned in {"", "N/A", "n/a", "nan", "NaN", "None", "none"}:
            return default
        try:
            numeric = float(cleaned)
            return default if np.isnan(numeric) else numeric
        except ValueError:
            return default

    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        return default if np.isnan(numeric) else numeric

    try:
        numeric = float(value)
        return default if np.isnan(numeric) else numeric
    except (TypeError, ValueError):
        return default


def sanitize_missing_values(value: Any) -> Any:
    """Recursively convert NumPy/Python missing values to blank strings for CSV/output use."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return {k: sanitize_missing_values(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_missing_values(v) for v in value]
    if isinstance(value, np.ndarray):
        return [sanitize_missing_values(v) for v in value.tolist()]
    if isinstance(value, (float, np.floating)):
        return "" if np.isnan(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value
