import numpy as np
import pandas as pd
from pathlib import Path
from itertools import chain
import inspect
from shared.context import EXPECTED_COILS, EXPECTED_ORIENTATIONS


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

    def construct_df_for_task(self, task):
        coil_dfs = [self.construct_df_for_coil(task, coil) for coil in EXPECTED_COILS]
        blank_rows = [self.blank_row for _ in range(len(coil_dfs))]
        return pd.concat(chain.from_iterable(zip(coil_dfs, blank_rows)))

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
            snr_norm_pairs = (
                [
                    self.chained_get(
                        task,
                        coil,
                        orientation,
                        "measurement",
                        "snr by smoothing",
                        "measured",
                    ),
                    self.chained_get(
                        task,
                        coil,
                        orientation,
                        "measurement",
                        "snr by smoothing",
                        "normalised",
                    ),
                ]
                for orientation in EXPECTED_ORIENTATIONS
            )
            snr, normalised_snr = zip(*snr_norm_pairs)
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

            av_perc_diffs, cvs = map(
                list, zip(*[get_perc_diff_and_cv(lq) for lq in length_quadruplets])
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


# class DataFrameCreator:

#     def __init__(self):
#         self.df_width = len(EXPECTED_ORIENTATIONS) + 1
#         self.df_components = {
#             "title_rows": [
#                 pd.DataFrame(
#                     [task.upper()] + [np.nan for x in range(self.df_width - 1)]
#                 ).T
#                 for task in settings.TASKS_TO_RUN
#             ],
#             "blank_row": pd.DataFrame(
#                 np.nan, index=range(1), columns=range(self.df_width)
#             ),
#             "orientations_row": pd.DataFrame([np.nan] + config.EXPECTED_ORIENTATIONS).T,
#             "coil_rows": {
#                 coil: pd.DataFrame([coil] + [np.nan] * (self.df_width - 1)).T
#                 for coil in config.EXPECTED_COILS
#             },
#         }

#     def run(self, excel_path: str) -> pd.DataFrame:
#         task_dataframes = {
#             task: pd.concat(
#                 self._list_with_delimiter(
#                     lst=[
#                         getattr(self, f"_populate_{task}")(
#                             modular_df_coil=self._construct_modular_df_for_coil(
#                                 coil=coil
#                             ),
#                             data=coil_dict,
#                         )
#                         for coil, coil_dict in task_dict.items()
#                     ],
#                     delim=self.df_components["blank_row"],
#                 ),
#                 ignore_index=True,
#             )
#             for task, task_dict in self.outer.results.items()
#         }

#         zipped = zip(
#             self.df_components["title_rows"],
#             list(task_dataframes.values()),
#             (self.df_components["blank_row"] for task in settings.TASKS_TO_RUN),
#         )

#         self.master_df = pd.concat(
#             [elem for triplet in zipped for elem in triplet], ignore_index=True
#         )
#         self.master_df.to_excel(
#             excel_path, header=False, index=False, sheet_name="Sheet1"
#         )

#     def _construct_modular_df_for_coil(self, coil: str):
#         modular_df_coil = pd.concat(
#             [
#                 self.df_components["coil_rows"][coil],
#                 self.df_components["orientations_row"],
#             ],
#             ignore_index=True,
#         )
#         return modular_df_coil

#     @staticmethod
#     def _populate_slice_thickness(modular_df_coil: pd.DataFrame, data: dict):
#         slice_thicknesses = [
#             data.get(orientation, {})
#             .get("measurement", {})
#             .get("slice width mm", "N/A")
#             for orientation in modular_df_coil.iloc[1, 1:]
#         ]

#         perc_diff_to_set = [
#             (st - 5) / 5 * 100 if isinstance(st, (int, float)) else "N/A"
#             for st in slice_thicknesses
#         ]

#         to_add = pd.DataFrame(
#             [
#                 ["Slice Thickness (mm)"] + slice_thicknesses,
#                 ["% Diff to set (5mm)"] + perc_diff_to_set,
#             ]
#         )

#         return pd.concat([modular_df_coil, to_add], ignore_index=True)

#     @staticmethod
#     def _populate_snr(modular_df_coil: pd.DataFrame, data: dict):
#         measured_and_normalised_snrs = [
#             (
#                 data.get(orientation, {})
#                 .get("measurement", {})
#                 .get("snr by smoothing", {})
#                 .get("measured", "N/A"),
#                 data.get(orientation, {})
#                 .get("measurement", {})
#                 .get("snr by smoothing", {})
#                 .get("normalised", "N/A"),
#             )
#             for orientation in modular_df_coil.iloc[1, 1:]
#         ]
#         measured_snrs, normalised_snrs = zip(*measured_and_normalised_snrs)

#         to_add = pd.DataFrame(
#             [
#                 ["Image SNR"] + list(measured_snrs),
#                 ["Normalised SNR"] + list(normalised_snrs),
#             ]
#         )

#         return pd.concat([modular_df_coil, to_add], ignore_index=True)

#     @staticmethod
#     def _populate_geometric_accuracy(modular_df_coil: pd.DataFrame, data: dict):
#         true_length = 173

#         def get_perc_diff_and_cv(lengths):
#             lengths = list(lengths)
#             perc_differences = [(1 - length / true_length) * 100 for length in lengths]
#             av_perc_diff = np.mean(perc_differences)
#             cv = np.std(lengths) / np.mean(lengths) * 100
#             return av_perc_diff, cv

#         lengths = [
#             data.get(orientation, {}).get("measurement", {}).values()
#             for orientation in modular_df_coil.iloc[1, 1:]
#         ]

#         perc_diffs_and_cvs = [
#             (get_perc_diff_and_cv(quad) if len(quad) != 0 else ["N/A"] * 2)
#             for quad in lengths
#         ]
#         perc_diffs, cvs = zip(*perc_diffs_and_cvs)

#         to_add = pd.DataFrame(
#             [["% Distortion"] + list(cvs), ["% Diff to actual"] + list(perc_diffs)]
#         )

#         return pd.concat([modular_df_coil, to_add], ignore_index=True)

#     @staticmethod
#     def _populate_uniformity(modular_df_coil: pd.DataFrame, data: dict):
#         uniformities = [
#             data.get(orientation, {})
#             .get("measurement", {})
#             .get("integral uniformity %", "N/A")
#             for orientation in modular_df_coil.iloc[1, 1:]
#         ]

#         to_add = pd.DataFrame(["% Integral Uniformity"] + uniformities).T

#         return pd.concat([modular_df_coil, to_add], ignore_index=True)

#     @staticmethod
#     def _populate_spatial_resolution(modular_df_coil: pd.DataFrame, data: dict):
#         raise NotImplementedError("Spatial resolution not yet implemented!")

#     @staticmethod
#     def _list_with_delimiter(lst: list, delim: any):
#         delimited_list = [
#             elem for pair in zip(lst, [delim] * (len(lst) - 1)) for elem in pair
#         ] + [lst[-1]]

#         return delimited_list
