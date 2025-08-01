"""
log_constructor.py

Constructs an excel log using pandas, the passed log path and the passed Hazen job running results.

Written by Nathan Crossley, 2025.
"""

from pathlib import Path
from itertools import chain
import inspect

import numpy as np
import pandas as pd

from src.backend.utils import chained_get

from src.shared.context import EXPECTED_COILS, EXPECTED_ORIENTATIONS
from src.shared.queueing import get_queue, QueueTrigger


class LogConstructor:
    """Class to construct excel log from taskrunner results."""

    def __init__(self, results: dict, log_path: Path):
        """Initialises LogConstructor. Determines various
        instance attributes for convenience.

        Args:
            results (dict): Results output from Hazen job running.
            log_path (Path): Path to save log to.
        """

        # define width of pandas df based on number of orientations
        self.width_df = len(EXPECTED_ORIENTATIONS) + 1

        # define a blank row
        self.blank_row = self.make_row(np.nan)

        # define a row for the different orientations
        self.orientations_header = self.make_row(np.nan, EXPECTED_ORIENTATIONS)

        # store results and log path for convenience
        self.results = results
        self.log_path = log_path

    def run(self):
        """Constructs a pd dataframe and saves it to an Excel file."""
        # Get list of relevant dataframes for each task.
        tasks = list(self.results.keys())
        task_headers = [self.make_row(task.upper()) for task in tasks]
        task_dfs = [self.construct_df_for_task(task) for task in tasks]
        blank_rows = [self.blank_row for _ in range(len(tasks))]

        # Interleave the task headers, dataframes, and blank rows and concat to form final Excel spreadsheet.
        master_df = pd.concat(
            chain.from_iterable(zip(task_headers, task_dfs, blank_rows))
        )
        master_df.to_excel(
            self.log_path, header=False, index=False, sheet_name="Sheet1"
        )

        # send app quitting trigger via queue.
        get_queue().put(QueueTrigger("QUIT_APPLICATION", master_df))

    def construct_df_for_task(self, task: str) -> pd.DataFrame:
        """Constructs a DataFrame for a specific Hazen task.

        Args:
            task (str): Task to construct DataFrame for.

        Returns:
            pd.DataFrame: Constructed DataFrame for the task.
        """
        coil_dfs = [self.construct_df_for_coil(task, coil) for coil in EXPECTED_COILS]
        blank_rows = [self.blank_row for _ in range(len(coil_dfs))]

        return pd.concat(list(chain.from_iterable(zip(coil_dfs, blank_rows)))[:-1])

    def construct_df_for_coil(self, task: str, coil: str) -> pd.DataFrame:
        """Constructs a DataFrame for a specific coil and task.
        Task-specific data is retrieved here.

        Args:
            task (str): Task to construct DataFrame for.
            coil (str): Coil to construct DataFrame for.

        Returns:
            pd.DataFrame: Constructed DataFrame for the coil and task.
        """

        # Get task specific data and convert to list if it's a DataFrame.
        # This is required so can unpack list properly in concat operation.
        task_specific_data = self.get_task_specific_data(task, coil)
        if isinstance(task_specific_data, pd.DataFrame):
            task_specific_data = [task_specific_data]

        to_concat = [
            self.make_row(coil.upper()),
            self.orientations_header,
            *task_specific_data,
        ]
        return pd.concat(to_concat)

    def get_task_specific_data(self, task: str, coil: str) -> list[pd.DataFrame]:
        """Retrieves task-specific data for a given task and coil from the results dict.
        Additional task-specific results are calculated here.

        Args:
            task (str): Task to retrieve data for.
            coil (str): Coil to retrieve data for.

        Returns:
            list[pd.DataFrame]: Dataframes containing task-specific data.
        """
        if task == "Slice Thickness":
            slice_thicknesses = [
                chained_get(
                    self.results,
                    task,
                    coil,
                    orientation,
                    "measurement",
                    "slice width mm",
                )
                for orientation in EXPECTED_ORIENTATIONS
            ]
            perc_diff_to_set = [
                (st - 5) / 5 * 100 if isinstance(st, (int, float)) else "N/A"
                for st in slice_thicknesses
            ]
            slice_thicknesses = self.make_row("Slice Thickness (mm)", slice_thicknesses)
            perc_diff_to_set = self.make_row("% Diff to set (5mm)", perc_diff_to_set)
            return slice_thicknesses, perc_diff_to_set

        elif task == "SNR":

            def pull_snr_values(
                smoothing: bool = True,
            ) -> tuple[list[float], list[float]]:
                """Gets the values for SNR for the specific coil.

                Args:
                    smoothing (bool, optional): flag for disabling SNR by subtraction . Defaults to True.

                Returns:
                    tuple[list[float], list[float]]: Lists of SNR and normalised SNR values.
                """
                # get pairs of SNR and normalised SNR
                snr_norm_pairs = (
                    [
                        chained_get(
                            self.results,
                            task,
                            coil,
                            orientation,
                            "measurement",
                            f"snr by {'smoothing' if smoothing else 'subtraction'}",
                            "measured",
                        ),
                        chained_get(
                            self.results,
                            task,
                            coil,
                            orientation,
                            "measurement",
                            f"snr by {'smoothing' if smoothing else 'subtraction'}",
                            "normalised",
                        ),
                    ]
                    for orientation in EXPECTED_ORIENTATIONS
                )
                snr, normalised_snr = zip(*snr_norm_pairs)
                return list(snr), list(normalised_snr)

            # Try to pull snr values using smoothing key
            snr, normalised_snr = pull_snr_values()

            # If all values are N/A, try to pull using subtraction key
            if all(
                [
                    x == inspect.signature(chained_get).parameters["default"].default
                    for x in snr + normalised_snr
                ]
            ):
                snr, normalised_snr = pull_snr_values(smoothing=False)

            snr = self.make_row("Image SNR", snr)
            normalised_snr = self.make_row("Normalised SNR", normalised_snr)
            return snr, normalised_snr

        elif task == "Geometric Accuracy":
            # Get quadruplets of lengths for each orientation for the specific coil.
            length_quadruplets = [
                chained_get(
                    self.results,
                    task,
                    coil,
                    orientation,
                    "measurement",
                    default=inspect.signature(chained_get)
                    .parameters["default"]
                    .default,
                )
                for orientation in EXPECTED_ORIENTATIONS
            ]

            def get_perc_diff_and_cv(length_quadruplet: dict) -> tuple[float]:
                """Get reported statistics from length quadruplet dict.

                Args:
                    length_quadruplet (dict): Dictionary of lengths.

                Returns:
                    tuple[float]: tuple of average percentage difference and coefficient of variation.
                """

                try:
                    true_length = 173
                    length_quadruplet = list(length_quadruplet.values())
                    perc_differences = [
                        (1 - length / true_length) * 100 for length in length_quadruplet
                    ]
                    av_perc_diff = np.mean(perc_differences)
                    cv = np.std(length_quadruplet) / np.mean(length_quadruplet) * 100
                    return av_perc_diff, cv
                except:
                    return [
                        inspect.signature(chained_get).parameters["default"].default
                    ] * 2

            av_perc_diffs, cvs = zip(
                *[get_perc_diff_and_cv(lq) for lq in length_quadruplets]
            )

            av_perc_diffs = self.make_row("% Distortion", list(av_perc_diffs))
            cvs = self.make_row("% Diff to actual", list(cvs))
            return av_perc_diffs, cvs

        elif task == "Uniformity":
            # get uniformity values for specific coil
            uniformity = [
                chained_get(
                    self.results,
                    task,
                    coil,
                    orientation,
                    "measurement",
                    "integral uniformity %",
                )
                for orientation in EXPECTED_ORIENTATIONS
            ]
            uniformity = self.make_row("% Integral Uniformity", uniformity)
            return uniformity

        elif task == "Spatial Resolution":
            # get mtf50 values for specific coil
            mtf50 = [
                chained_get(
                    self.results, task, coil, orientation, "measurement", "mtf50"
                )
                for orientation in EXPECTED_ORIENTATIONS
            ]

            # calculate spatial resolution from mtf50
            spatial_res = [
                (
                    1 / mtf
                    if isinstance(mtf, (int, float))
                    else inspect.signature(chained_get).parameters["default"].default
                )
                for mtf in mtf50
            ]
            spatial_res = self.make_row("Spatial Resolution", spatial_res)
            return spatial_res

    def make_row(self, label: str, values: list = None) -> pd.DataFrame:
        """Returns a DataFrame with a single row containing passed label and values.
        If values is None, a blank row is used in place of values.

        Args:
            label (str): Label for the DataFrame.
            values (list, optional): Values to display to the right of label. Defaults to None.

        Returns:
            pd.DataFrame: Dataframe containing label and values.
        """

        if not isinstance(values, (type(None), list)):
            raise TypeError("values attr should be list.")
        if values is None:
            values = [np.nan for _ in range(self.width_df - 1)]
        return pd.DataFrame([label] + values).T
