import numpy as np
import pandas as pd
from pathlib import Path
from itertools import chain
import inspect
from shared.context import EXPECTED_COILS, EXPECTED_ORIENTATIONS
from shared.queueing import get_queue


class DataFrameConstructor:

    def __init__(self, results, excel_path: Path):
        self.width_df = len(EXPECTED_ORIENTATIONS) + 1
        self.blank_row = self.make_row(np.nan)
        self.orientations_header = self.make_row(np.nan, EXPECTED_ORIENTATIONS)
        self.results = results
        self.excel_path = excel_path

    def run(self):
        tasks = {k for k in self.results.keys()}
        task_headers = [self.make_row(task.upper()) for task in tasks]
        task_dfs = [self.construct_df_for_task(task) for task in tasks]
        blank_rows = [self.blank_row for _ in range(len(tasks))]

        master_df = pd.concat(
            chain.from_iterable(zip(task_headers, task_dfs, blank_rows))
        )
        master_df.to_excel(
            self.excel_path, header=False, index=False, sheet_name="Sheet1"
        )
        get_queue().put(("TASK_COMPLETE", "DATAFRAME_CONSTRUCTED", master_df))

    def construct_df_for_task(self, task):
        coil_dfs = [self.construct_df_for_coil(task, coil) for coil in EXPECTED_COILS]
        blank_rows = [self.blank_row for _ in range(len(coil_dfs))]
        return pd.concat(list(chain.from_iterable(zip(coil_dfs, blank_rows)))[:-1])

    def construct_df_for_coil(self, task, coil):
        task_specific_data = self.get_task_specific_data(task, coil)
        if isinstance(task_specific_data, pd.DataFrame):
            task_specific_data = [task_specific_data]

        to_concat = [
            self.make_row(coil.upper()),
            self.orientations_header,
            *task_specific_data,
        ]
        return pd.concat(to_concat)

    def get_task_specific_data(self, task, coil):
        if task == "Slice Thickness":
            slice_thicknesses = [
                self.chained_get(
                    task, coil, orientation, "measurement", "slice width mm"
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

            def pull_snr_values(smoothing: bool = True):
                snr_norm_pairs = (
                    [
                        self.chained_get(
                            task,
                            coil,
                            orientation,
                            "measurement",
                            f"snr by {'smoothing' if smoothing else 'subtraction'}",
                            "measured",
                        ),
                        self.chained_get(
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
                    x
                    == inspect.signature(self.chained_get).parameters["default"].default
                    for x in snr + normalised_snr
                ]
            ):
                snr, normalised_snr = pull_snr_values(smoothing=False)

            snr = self.make_row("Image SNR", snr)
            normalised_snr = self.make_row("Normalised SNR", normalised_snr)
            return snr, normalised_snr

        elif task == "Geometric Accuracy":
            length_quadruplets = [
                self.chained_get(
                    task,
                    coil,
                    orientation,
                    "measurement",
                    default=inspect.signature(self.chained_get)
                    .parameters["default"]
                    .default,
                )
                for orientation in EXPECTED_ORIENTATIONS
            ]

            def get_perc_diff_and_cv(length_quadruplet):
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
                        inspect.signature(self.chained_get)
                        .parameters["default"]
                        .default
                    ] * 2

            av_perc_diffs, cvs = zip(
                *[get_perc_diff_and_cv(lq) for lq in length_quadruplets]
            )

            av_perc_diffs = self.make_row("% Distortion", list(av_perc_diffs))
            cvs = self.make_row("% Diff to actual", list(cvs))
            return av_perc_diffs, cvs

        elif task == "Uniformity":
            uniformity = [
                self.chained_get(
                    task, coil, orientation, "measurement", "integral uniformity %"
                )
                for orientation in EXPECTED_ORIENTATIONS
            ]
            uniformity = self.make_row("% Integral Uniformity", uniformity)
            return uniformity

    def make_row(self, label: str, values: list = None):
        if not isinstance(values, (type(None), list)):
            raise TypeError("values attr should be list.")
        if values is None:
            values = [np.nan for _ in range(self.width_df - 1)]
        return pd.DataFrame([label] + values).T

    def chained_get(self, *keys, default="N/A"):
        d = self.results.copy()
        for key in keys:
            if isinstance(d, dict):
                d = d.get(key, None)
                if d is None:
                    return default
        return d
