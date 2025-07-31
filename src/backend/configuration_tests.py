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

    # get test dcm as first dcm from first input subdirectory
    test_dir = in_subdirs[0]
    test_files = [x for x in test_dir.iterdir() if x.is_file()]
    test_dcm = test_files[0]

    # get manufacturer tag and string match to expected manufacturer
    metadata = pydicom.dcmread(test_dcm, stop_before_pixels=True)
    manufacturer_tag = metadata.get("Manufacturer")
    manufacturer = substring_matcher(manufacturer_tag, IMPLEMENTED_MANUFACTURERS)

    # initialised lists for storing problematic subdirectories
    subdirs_count_err = []
    subdirs_helper_missing_err = []

    # iterate over each input subdir
    for in_subdir in in_subdirs:

        # if SNR task wants to be run by user and if manufacturer is Philips, perform this check
        if "SNR" in tasks_to_run and manufacturer == "Philips":

            # define path to helper data set
            helper_data_set = in_subdir / "helper_data_set"

            # if helper data set exists, add subdir to correct error list if number of dcms not 11 (sorting has failed somehow)
            if helper_data_set.exists():
                num_dcms = len(list(helper_data_set.iterdir()))

                if num_dcms != 11:
                    subdirs_count_err.append(in_subdir.name)

            # else add subdir to error list for missing helper data sets
            else:
                subdirs_helper_missing_err.append(in_subdir.name)

    # construct phrase informing user that some helper data sets are missing, where they are required.
    if len(subdirs_helper_missing_err) != 0:
        plural_helper_missing = len(subdirs_helper_missing_err) > 1
        helper_missing_phrase = f"Tried to implement SNR by subtraction for director{'ies' if plural_helper_missing else 'y'} {', '.join(subdirs_helper_missing_err)} but could not find helper data set{'s' if plural_helper_missing else ''}!"
    else:
        helper_missing_phrase = ""

    # construct phrase informing that that number of dcms in various helper data sets are missing, where 11 is expected for accuracy
    if len(subdirs_count_err) != 0:
        plural_count_err = len(subdirs_count_err) > 1
        count_err_phrase = f"Tried to implement SNR by subtraction for director{'ies' if plural_count_err else 'y'} {', '.join(subdirs_count_err)} but too many dcms found in helper data set{'s' if plural_count_err else ''}!"
    else:
        count_err_phrase = ""

    # if any errors need to be raised, add phrase for advice
    if helper_missing_phrase or count_err_phrase:
        advice_phrase = "Before browsing for input dcm directory, please ensure that helper data set dcms have the same SeriesDescription tag as the main ones, and that repeat acquisitions for other reasons are labelled with a different SeriesDescription tag."
    else:
        advice_phrase = ""

    # collate phrases into a list with truthy check
    phrases = [helper_missing_phrase, count_err_phrase, advice_phrase]
    phrases = [x for x in phrases if x]

    # if errors need to be raised, show using messagebox and then return True to indicate that there are errors
    if len(phrases) != 0:
        errors = f"{"\n\n".join(phrases)}"
        tk.messagebox.showerror("Error", errors)
        return True

    # otherwise return False
    else:
        return False
