"""
configuration_tests.py

This script contains backend functions to test the configuration settings of AutoQC_ACR before the actual Hazen jobs
start running. This ensures that results are accurate and unexpected erros are not thrown mid task runtime.

Written by Nathan Crossley 2025.

"""

import pydicom
from pathlib import Path
import tkinter as tk

from src.backend.utils import substring_matcher
from src.shared.context import IMPLEMENTED_MANUFACTURERS
from src.backend.dcm_sorter import find_philips_helper_dir


def file_structure_problems_exist(
    in_subdirs: list[Path], tasks_to_run: list[str]
) -> str:
    """Analyses the file structure of the input subdirectories and
    returns a string documenting those errors. The error string can
    then be displayed to the user by a GUI window and the actual task
    running process can then be cancelled.

    Args:
        in_subdirs (list[Path]): List of input subdirectories taken from GUI configuration page.
        tasks_to_run (list[str]): List of Hazen tasks to run, taken from GUI configuration page.

    Returns:
        str: String error to be displayed to the user.
    """

    test_dir = in_subdirs[0]
    test_files = [x for x in test_dir.iterdir() if x.is_file()]
    test_dcm = test_files[0]

    metadata = pydicom.dcmread(test_dcm, stop_before_pixels=True)
    manufacturer_tag = metadata.get("Manufacturer")
    manufacturer = substring_matcher(manufacturer_tag, IMPLEMENTED_MANUFACTURERS)

    subdirs_count_err = []
    subdirs_helper_missing_err = []

    for in_subdir in in_subdirs:
        if "SNR" in tasks_to_run and manufacturer == "Philips":
            helper_data_set = in_subdir / "helper_data_set"
            if helper_data_set.exists():
                num_dcms = len(list(helper_data_set.iterdir()))
                if num_dcms != 11:
                    subdirs_count_err.append(in_subdir.name)
            else:
                helper_dir = find_philips_helper_dir(in_subdir)
                if helper_dir is None:
                    subdirs_helper_missing_err.append(in_subdir.name)

    if len(subdirs_helper_missing_err) != 0:
        plural_helper_missing = len(subdirs_helper_missing_err) > 1
        helper_missing_phrase = f"Tried to implement SNR by subtraction for director{'ies' if plural_helper_missing else 'y'} {', '.join(subdirs_helper_missing_err)} but could not find helper data set{'s' if plural_helper_missing else ''}!"
    else:
        helper_missing_phrase = ""

    if len(subdirs_count_err) != 0:
        plural_count_err = len(subdirs_count_err) > 1
        count_err_phrase = f"Tried to implement SNR by subtraction for director{'ies' if plural_count_err else 'y'} {', '.join(subdirs_count_err)} but too many dcms found in helper data set{'s' if plural_count_err else ''}!"
    else:
        count_err_phrase = ""

    if helper_missing_phrase or count_err_phrase:
        advice_phrase = "Before browsing for input dcm directory, please ensure that helper data set dcms are the same acquisition type as the main ones, and that true reacquisitions or other repeat acquisitions are labelled with a different SeriesDescription tag or a non-numeric appendix."
    else:
        advice_phrase = ""

    phrases = [helper_missing_phrase, count_err_phrase, advice_phrase]
    phrases = [x for x in phrases if x]

    if len(phrases) != 0:
        errors = "\n\n".join(phrases)
        tk.messagebox.showerror("Error", errors)
        return True

    return False
