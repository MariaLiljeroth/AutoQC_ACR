from copy import deepcopy
import numbers
import sys
import numpy as np
from pathlib import Path
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
import numbers

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    ListFlowable,
    ListItem,
)
from copy import deepcopy
import numbers
import sys
import numpy as np
from pathlib import Path
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors

from src.backend.utils import chained_get
from src.shared.context import EXPECTED_ORIENTATIONS


class ReportGenerator:

    REPORT_NAME = "AQA_Report.pdf"
    STYLES = getSampleStyleSheet()

    if getattr(sys, "frozen", False):
        LOGO_PATH = str((Path(sys.executable).parent / "_internal/assets/royal_surrey_logo.png").resolve())
    else:
        LOGO_PATH = "src/backend/assets/royal_surrey_logo.png"

    DEPARTMENT_NAME = "Regional Radiation Protection Service"
    DEPARTMENT_INFO = "Research & Oncology Suite, Royal Surrey County Hospital Guildford Surrey GU2 7XX Tel: 01483 408395 Email:rsc-tr.RadProt@nhs.net"
    Title_header = "Appendix A: Detailed Results of the Survey "
    TABLE_STYLE_DEFAULT = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 9),
            ("BACKGROUND", (0, 1), (-1, -1), colors.white),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]
    )

    DEEP_KEYS = {
        "Slice Thickness": [["measurement", "slice width mm"]],
        "SNR": [["measurement", "snr by smoothing", "measured"]],
        "Geometric Accuracy": [
            ["measurement", "Horizontal distance"],
            ["measurement", "Vertical distance"],
            ["measurement", "Diagonal distance SE"],
            ["measurement", "Diagonal distance SW"],
        ],
        "Uniformity": [["measurement", "integral uniformity %"]],
        "Spatial Resolution": [["measurement", "mtf50"]],
    }

    BULLET_KWARGS = {
        "bulletType": "bullet",
        "bulletFontName": "Helvetica",
        "bulletColor": colors.black,
        "bulletFontSize": 14,
        "bulletIndent": 0,
    }

    SUBTITLES = {
        "Slice Thickness": "This measurement gives an indication as to the overall performance of the imaging gradients and the homogeneity of the main magnetic field. <br/> In accordance with IPEM report 112, variations of less than 0.7 mm for a prescribed slice thickness of 5 mm is acceptable. If measurements fall outside of this tolerance, this implies an issue with:",
        "SNR": "This measurement gives an indication as to the overall performance of the scanner. <br/> In accordance with IPEM report 112, variations of less than 10 % are considered acceptable. <br/><br/> If measurements fall outside of this tolerance, this implies either an inconsistency in acquisition parameters, uneven loading or a genuine issue with the scanner. If determined to be a genuine issue with the scanner, a decrease in SNR can be indicative of: ",
        "Geometric Accuracy": "This measurement gives an indication, primarily, as to the performance of the imaging gradients. Poor geometric accuracy will cause image distortions such as edge warping. This measurement reports the percentage deviation from the actual phantom diameter as well as the absolute deviation in mm. In accordance with AAPM guidelines, the percentage linearity error should not exceed 2 %. ACR guidelines recommend that the absolute deviation should not exceed 2 mm. A increase in percentage linearity error is indicative of: ",
        "Uniformity": "This measurement gives an indication as to the overall performance of the imaging gradients and the homogeneity of the main magnetic field. <br/> In accordance with the ACR Phantom Test Guidance PIU should be greater than or equal to 87.5% for MRI systems with field strengths less than 3 Tesla. PIU should be greater than or equal to 82.0% for MRI systems with field strength of 3 Tesla.If measurements fall outside of this tolerance, this implies an issue with: ",
        "Spatial Resolution": "This measurement gives an indication as to the ability of the scanner to image subtle anatomical features by assessing the in-plane resolution. It is important to note that deviations to this may be caused by partial voluming due to non-isotropic voxels although this should cause no more than a small deviation to the results. This test estimates the modulation transfer function and the measured effective resolution. There are no official tolerance intervals for this measurement and the suggestion is to follow this over time and be observant to any unexpected changes. As an inofficial rule of thumb, less than half a pixel size (0.98 +/-0.49 mm) deviation is considered acceptable at clinical field strengths at or less than 3T. An increase in this measurement may imply:",
    }

    BULLET_POINTS = {
        "Slice Thickness": ["Imaging gradient non-linearity", "Poor B0 shim"],
        "SNR": [
            "Increased noise levels due to e.g interference, coil element coupling or broken coil element",
            "Decreased signal levels due to poor B0 shim, issues with transmit gain or overall degradation of the system",
        ],
        "Geometric Accuracy": ["Imaging gradient non-linearity", "Poor B0 shim"],
        "Uniformity": ["Poor B0 shim", "Poor B1 shim", "Coil element failure/cross-talk", "Uneven or incomplete loading"],
        "Spatial Resolution": ["Imaging gradient non-linearity", "Poor B0 shim"],
    }

    COIL_DESCRIPTIONS = {
        "IB": "Inbuilt Transmit-Receive Coil",
        "HN": "Head & Neck Coil",
        "Flex": "Flexible Phased Array Anterior Coil",
    }

    TRUE_SLICE_THICKNESS = 5  # mm
    TRUE_PHANTOM_DIAMETER = 173  # mm

    COLUMNS_TO_CHECK = {
        "Slice Thickness": [(-1, 0.7)],
        "SNR": [(-1, None)],
        "Geometric Accuracy": [(-2, 2), (-1, 2)],
        "Uniformity": [(-1, None)],
        "Spatial Resolution": [(-1, None)],
    }

    def __init__(self, results, baselines, field_strength, out_dir, coils: list[str] | None = None):
        self.results = results
        self.baselines = baselines
        self.field_strength = field_strength
        self.out_dir = out_dir
        self.story = []

        self.thresholds = {
            "Slice Thickness": 0.7,
            "SNR": 10,
            "Geometric Accuracy": 2,
            "Uniformity": 82 if field_strength == 3 else 87.5,
            "Spatial Resolution": None,
        }

        # default coils list will be populated by caller; fallback to shared context expected coils
        from src.shared.context import EXPECTED_COILS
        self.coils = coils or EXPECTED_COILS
        # dynamic mapping from coil id to row index based on provided coils
        self.COIL_ROW_MAP = {coil: i for i, coil in enumerate(self.coils)}

    def run(self):
        self.add_header()
        for task in self.results:
            self.add_section(task)
        report_path = self.out_dir / self.REPORT_NAME
        doc = SimpleDocTemplate(str(report_path.resolve()), pagesize=letter)
        doc.build(self.story)
        print("✅ PDF created !")

    def add_header(self):
        title = Paragraph(f'<para align="left">{self.Title_header}</para>', self.STYLES["Heading1"])
        self.story.append(title)
        self.story.append(Spacer(1, 28))

    def add_section(self, task):
        matrix = self.extract_task_matrix(task)
        table_data = self.get_table_data(task, matrix)
        self.build_metric_section(task, table_data)

    def extract_task_matrix(self, task):
    # """
    # Extracts a 3D matrix for a given task.
    
    # Dimensions: [num_coils, num_orientations, num_metrics]
    
    # Special handling for SNR: prefer subtraction, fallback to smoothing.
    # """
        if task == "SNR":
            num_metrics = 1
        else:
            deep_keys = self.DEEP_KEYS[task]
            num_metrics = len(deep_keys)

        data = np.full((len(self.coils), len(EXPECTED_ORIENTATIONS), num_metrics), np.nan)

        for i, coil in enumerate(self.coils):
            coil_dict = self.results.get(task, {}).get(coil, {})
            for j, orientation in enumerate(EXPECTED_ORIENTATIONS):
                orient_dict = coil_dict.get(orientation, {})

                if task == "SNR":
                    # Special-case: prefer subtraction if available, else smoothing
                    meas = orient_dict.get("measurement", {})
                    val = None
                    if "snr by subtraction" in meas:
                        val = meas["snr by subtraction"].get("measured")
                    elif "snr by smoothing" in meas:
                        val = meas["snr by smoothing"].get("measured")

                    try:
                        data[i, j, 0] = float(val)
                    except (TypeError, ValueError):
                        data[i, j, 0] = np.nan

                else:
                    # Generic case for other tasks
                    for k, metric_keys in enumerate(self.DEEP_KEYS[task]):
                        val = orient_dict
                        for key in metric_keys:
                            if isinstance(val, dict):
                                val = val.get(key, None)
                            else:
                                val = None
                                break
                        if isinstance(val, dict) and "measured" in val:
                            val = val["measured"]

                        try:
                            data[i, j, k] = float(val)
                        except (TypeError, ValueError):
                            data[i, j, k] = np.nan
        return data
        
    # def extract_task_matrix(self, task):
    #     if task == "SNR":
    #         meas = orient_dict.get("measurement", {})
    #         print(f"DEBUG meas keys: {list(meas.keys())}")  # <-- show all keys
    #         val = None
    #         if "snr by subtraction" in meas:
    #             val = meas["snr by subtraction"].get("measured")
    #         elif "snr by smoothing" in meas:
    #             val = meas["snr by smoothing"].get("measured")
    #         print(f"DEBUG val for {coil} {orientation}: {val}")


    #         num_metrics = 1
    #     else:
    #         deep_keys = self.DEEP_KEYS[task]
    #         num_metrics = len(deep_keys)

    #     data = np.full((len(EXPECTED_COILS), len(EXPECTED_ORIENTATIONS), num_metrics), np.nan)

    #     for i, coil in enumerate(EXPECTED_COILS):
    #         coil_dict = self.results.get(task, {}).get(coil, {})
    #         for j, orientation in enumerate(EXPECTED_ORIENTATIONS):
    #             orient_dict = coil_dict.get(orientation, {})

    #             if task == "SNR":
    #                 # Special-case: prefer subtraction if available, else smoothing
    #                 meas = orient_dict.get("measurement", {})
    #                 val = None
    #                 if "snr by subtraction" in meas:
    #                     val = meas["snr by subtraction"].get("measured")
    #                 elif "snr by smoothing" in meas:
    #                     val = meas["snr by smoothing"].get("measured")

    #                 try:
    #                     data[i, j, 0] = float(val)
    #                 except (TypeError, ValueError):
    #                     data[i, j, 0] = np.nan

    #             else:
    #                 # Generic case for other tasks
    #                 for k, metric_keys in enumerate(self.DEEP_KEYS[task]):
    #                     val = orient_dict
    #                     for key in metric_keys:
    #                         if isinstance(val, dict):
    #                             val = val.get(key, None)
    #                         else:
    #                             val = None
    #                             break
    #                     if isinstance(val, dict) and "measured" in val:
    #                         val = val["measured"]

    #                     try:
    #                         data[i, j, k] = float(val)
    #                     except (TypeError, ValueError):
    #                         data[i, j, k] = np.nan
    #     return data

    # def extract_task_matrix(self, task):
    #     deep_keys = self.DEEP_KEYS[task]
    #     num_metrics = len(deep_keys)
    #     data = np.full((len(EXPECTED_COILS), len(EXPECTED_ORIENTATIONS), num_metrics), np.nan)

    #     for i, coil in enumerate(EXPECTED_COILS):
    #         coil_dict = self.results.get(task, {}).get(coil, {})
    #         for j, orientation in enumerate(EXPECTED_ORIENTATIONS):
    #             orient_dict = coil_dict.get(orientation, {})
    #             for k, metric_keys in enumerate(deep_keys):
    #                 val = orient_dict
    #                 #debug statement
    #                 for key in metric_keys:
    #                     if isinstance(val, dict):
    #                         val = val.get(key, None)
    #                     else:
    #                         val = None
    #                         print("this is happening")
    #                         break
    #                 # If value is dict with "measured", extract
    #                 if isinstance(val, dict) and "measured" in val:
    #                     val = val["measured"]
                    
    #                 try:
    #                     data[i, j, k] = float(val)
    #                 except (TypeError, ValueError):
    #                     data[i, j, k] = np.nan
    #     return data

    def get_table_data(self, task, matrix):
        basic_table_header = self.get_table_header(task)
        table_data = [basic_table_header]
        numeric_copy = []

        for coil_id, row_idx in self.COIL_ROW_MAP.items():
            row_basic = self.get_basic_row_data(matrix, row_idx, coil_id)

            if task == "Slice Thickness":
                diff_from_actual = abs(row_basic[-1] - self.TRUE_SLICE_THICKNESS)
                row_extension = [diff_from_actual]
            elif task == "SNR":
                baseline = self.baselines.loc[task, coil_id]
                perc_diff = float(
                    self.calc_deviation_row_mean_from_val(matrix, row_idx, baseline, representation="percentage")
                )
                row_extension = [baseline, round(perc_diff, 2)]
            elif task == "Geometric Accuracy":
                deviation = float(
                    self.calc_deviation_row_mean_from_val(matrix, row_idx, self.TRUE_PHANTOM_DIAMETER, representation="absolute")
                )
                deviationP = float(
                    self.calc_deviation_row_mean_from_val(matrix, row_idx, self.TRUE_PHANTOM_DIAMETER, representation="percentage")
                )
                row_extension = [round(deviation, 2), round(deviationP, 2)]
            else:
                row_extension = []

            full_row = row_basic + row_extension
            table_data.append(full_row)

            numeric_copy.append([cell if isinstance(cell, numbers.Real) else None for cell in full_row])

        return table_data, numeric_copy

    @staticmethod
    def get_table_header(task):
        HEADER_EXTENSIONS = {
            "Slice Thickness": ["Deviation from prescribed (mm)"],
            "SNR": ["Baseline SNR", "Deviation from baseline (%)"],
            "Geometric Accuracy": ["Actual deviation (mm)", "Percentage linearity error (%)"],
            "Uniformity": [],
            "Spatial Resolution": [],
        }
        basic_table_header = [" ", "Axial", "Sagittal", "Coronal", f"Average {task}"]
        final_header = basic_table_header + HEADER_EXTENSIONS.get(task, [])
        return final_header

    def get_basic_row_data(self, matrix, row_idx, coil_id):
        matrix_row = matrix[row_idx, :3]
        if isinstance(matrix_row[0], (list, tuple, np.ndarray)):
            matrix_row = [np.mean(x) for x in matrix_row]
        coil_desc = self.COIL_DESCRIPTIONS.get(coil_id, coil_id)
        return [Paragraph(f"<b>{coil_desc}</b>"), *matrix_row, np.nanmean(matrix_row)]

    @staticmethod
    def calc_deviation_row_mean_from_val(matrix, row, reference_val, representation="absolute"):
        row_average = np.nanmean(matrix[row, :])
        deviation = row_average - reference_val
        if representation == "absolute":
            return deviation
        elif representation == "percentage":
            return np.abs(deviation / reference_val) * 100
        else:
            raise ValueError("representation must be 'absolute' or 'percentage'")

    def build_metric_section(self, task, table_data_numeric_tuple):
        table_style = deepcopy(self.TABLE_STYLE_DEFAULT)
        table_data, numeric_copy = table_data_numeric_tuple

        wrap_style = ParagraphStyle('wrap', fontName='Helvetica', fontSize=9, leading=12)
        wrap_style_header = ParagraphStyle('wrap_header', fontName='Helvetica-Bold', fontSize=9, leading=12, textColor=colors.white)

        self.story.append(Paragraph(f"<font size=16><b>{task}</b></font>", self.STYLES["Normal"]))
        self.story.append(Spacer(1, 20))
        self.story.append(Paragraph(f"<font size=12>{self.SUBTITLES[task]}</font>", self.STYLES["Normal"]))
        self.story.append(Spacer(1, 12))

        bullet_list = ListFlowable([ListItem(Paragraph(b, self.STYLES["Normal"])) for b in self.BULLET_POINTS[task]], **self.BULLET_KWARGS)
        self.story.append(bullet_list)
        self.story.append(Spacer(1, 22))

        THRESHOLD_CONDITIONS = {"Slice Thickness": ">", "SNR": ">", "Geometric Accuracy": ">", "Uniformity": "<", "Spatial Resolution": ">"}
        thresholds_for_task = self.COLUMNS_TO_CHECK.get(task, [(-1, self.thresholds.get(task))])

        for row_idx in range(1, len(table_data)):
            for col_idx, threshold in thresholds_for_task:
                if threshold is None:
                    threshold = self.thresholds.get(task)
                if threshold is None:
                    continue
                try:
                    value = numeric_copy[row_idx - 1][col_idx]
                    if value is None:
                        continue
                    cell_coords = (col_idx, row_idx)
                    if (THRESHOLD_CONDITIONS[task] == ">" and abs(value) > threshold) or (THRESHOLD_CONDITIONS[task] == "<" and abs(value) < threshold):
                        table_style.add("BACKGROUND", cell_coords, cell_coords, colors.salmon)
                    else:
                        table_style.add("BACKGROUND", cell_coords, cell_coords, colors.lightgreen)
                except Exception:
                    continue

        display_table_data = []
        for row_idx, row in enumerate(table_data):
            formatted_row = []
            for i, cell in enumerate(row):
                if row_idx == 0:
                    formatted_row.append(Paragraph(str(cell), wrap_style_header))
                elif i == 0 and isinstance(cell, Paragraph):
                    formatted_row.append(cell)
                elif isinstance(cell, numbers.Real):
                    formatted_row.append(Paragraph(f"{cell:.2f}", wrap_style))
                else:
                    formatted_row.append(Paragraph(str(cell), wrap_style))
            display_table_data.append(formatted_row)

        PAGE_WIDTH, PAGE_HEIGHT = letter
        LEFT_MARGIN = RIGHT_MARGIN = 72
        TEXT_WIDTH = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
        num_cols = len(display_table_data[0])
        col_ratios = [0.25, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15] if num_cols == 7 else [1/num_cols]*num_cols
        col_widths = [TEXT_WIDTH * r for r in col_ratios]

        table = Table(display_table_data, colWidths=col_widths, hAlign='LEFT', repeatRows=1)
        table.setStyle(table_style)
        self.story.append(table)
        self.story.append(Spacer(1, 12))

# class ReportGenerator:

#     REPORT_NAME = "AQA_Report.pdf"
#     STYLES = getSampleStyleSheet()

#     if getattr(sys, "frozen", False):
#         LOGO_PATH = str(
#             (
#                 Path(sys.executable).parent / "_internal/assets/royal_surrey_logo.png"
#             ).resolve()
#         )
#     else:
#         LOGO_PATH = "src/backend/assets/royal_surrey_logo.png"

#     DEPARTMENT_NAME = "Regional Radiation Protection Service"
#     DEPARTMENT_INFO = "Research & Oncology Suite, Royal Surrey County Hospital Guildford Surrey GU2 7XX Tel: 01483 408395 Email:rsc-tr.RadProt@nhs.net"
#     Title_header = "Appendix A: Detailed Results of the Survey "
#     TABLE_STYLE_DEFAULT = TableStyle(
#         [
#             ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
#             ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
#             ("ALIGN", (0, 0), (-1, -1), "CENTER"),
#             ("FONTNAME", (0, 0), (-1, 0), "Helvetica"),
#             ("FONTSIZE", (0, 0), (-1, 0), 9),
#             ("BOTTOMPADDING", (0, 0), (-1, 0), 9),
#             ("BACKGROUND", (0, 1), (-1, -1), colors.white),
#             ("GRID", (0, 0), (-1, -1), 1, colors.black),
#         ]
#     )

#     DEEP_KEYS = {
#         "Slice Thickness": [["measurement", "slice width mm"]],
#         "SNR": [["measurement", "snr by smoothing", "measured"]],
#         "Geometric Accuracy": [
#             ["measurement", "Horizontal distance"],
#             ["measurement", "Vertical distance"],
#             ["measurement", "Diagonal distance SE"],
#             ["measurement", "Diagonal distance SW"],
#         ],
#         "Uniformity": [["measurement", "integral uniformity %"]],
#         "Spatial Resolution": [["measurement", "mtf50"]],
#     }

#     BULLET_KWARGS = {
#         "bulletType": "bullet",
#         "bulletFontName": "Helvetica",
#         "bulletColor": colors.black,
#         "bulletFontSize": 14,
#         "bulletIndent": 0,
#     }

#     SUBTITLES = {
#         "Slice Thickness": "This measurement gives an indication as to the overall performance of the imaging gradients and the homogeneity of the main magnetic field. <br/> In accordance with IPEM report 112, variations of less than 0.7 mm for a prescribed slice thickness of 5 mm is acceptable. If measurements fall outside of this tolerance, this implies an issue with:",
#         "SNR": "This measurement gives an indication as to the overall performance of the scanner. <br/> In accordance with IPEM report 112, variations of less than 10 % are considered acceptable. <br/><br/> If measurements fall outside of this tolerance, this implies either an inconsistency in acquisition parameters, uneven loading or a genuine issue with the scanner. If determined to be a genuine issue with the scanner, a decrease in SNR can be indicative of: ",
#         "Geometric Accuracy": "This measurement gives an indication, primarily, as to the performance of the imaging gradients. Poor geometric accuracy will cause image distortions such as edge warping. This measurement reports the percentage deviation from the actual phantom diameter as well as the absolute deviation in mm. In accordance with AAPM guidelines, the percentage linearity error should not exceed 2 %. ACR guidelines recommend that the absolute deviation should not exceed 2 mm. A increase in percentage linearity error is indicative of: ",
#         "Uniformity": "This measurement gives an indication as to the overall performance of the imaging gradients and the homogeneity of the main magnetic field. <br/> In accordance with the ACR Phantom Test Guidance PIU should be greater than or equal to 87.5% for MRI systems with field strengths less than 3 Tesla. PIU should be greater than or equal to 82.0% for MRI systems with field strength of 3 Tesla.If measurements fall outside of this tolerance, this implies an issue with: ",
#         "Spatial Resolution": "This measurement gives an indication as to the ability of the scanner to image subtle anatomical features by assessing the in-plane resolution. It is important to note that deviations to this may be caused by partial voluming due to non-isotropic voxels although this should cause no more than a small deviation to the results. This test estimates the modulation transfer function and the measured effective resolution. There are no official tolerance intervals for this measurement and the suggestion is to follow this over time and be observant to any unexpected changes. As an inofficial rule of thumb, less than half a pixel size (0.98 +/-0.49 mm) deviation is considered acceptable at clinical field strengths at or less than 3T. An increase in this measurement may imply:",
#     }

#     BULLET_POINTS = {
#         "Slice Thickness": ["Imaging gradient non-linearity", "Poor B0 shim"],
#         "SNR": [
#             "Increased noise levels due to e.g interference, coil element coupling or broken coil element",
#             "Decreased signal levels due to poor B0 shim, issues with transmit gain or overall degradation of the system",
#         ],
#         "Geometric Accuracy": ["Imaging gradient non-linearity", "Poor B0 shim"],
#         "Uniformity": [
#             "Poor B0 shim",
#             "Poor B1 shim",
#             "Coil element failure/cross-talk",
#             "Uneven or incomplete loading",
#         ],
#         "Spatial Resolution": ["Imaging gradient non-linearity", "Poor B0 shim"],
#     }

#     COIL_DESCRIPTIONS = {
#         "IB": "Inbuilt Transmit-Receive Coil",
#         "HN": "Head & Neck Coil",
#         "Flex": "Flexible Phased Array Anterior Coil",
#     }

#     COIL_ROW_MAP = {"IB": 0, "HN": 1, "Flex": 2}

#     TRUE_SLICE_THICKNESS = 5  # mm
#     TRUE_PHANTOM_DIAMETER = 173  # mm

#     # 🔹 Column + threshold config (per task)
#     COLUMNS_TO_CHECK = {
#         "Slice Thickness": [(-1, 0.7)],
#         # "SNR": [(-2, None), (-1, 10)],
#         "SNR": [(-1, None)],
#         "Geometric Accuracy": [(-2, 2), (-1, 2)],
#         "Uniformity": [(-1, None)],   # defer to self.thresholds
#         "Spatial Resolution": [(-1, None)],
#     }

#     def __init__(self, results, baselines, field_strength, out_dir):
#         self.results = results
#         self.baselines = baselines
#         self.field_strength = field_strength
#         self.out_dir = out_dir
#         self.story = []

#         self.thresholds = {
#             "Slice Thickness": 0.7,
#             "SNR": 10,
#             "Geometric Accuracy": 2,
#             "Uniformity": 82 if field_strength == 3 else 87.5,
#             "Spatial Resolution": None,
#         }

#     def run(self):
#         self.add_header()
#         for task in self.results:
#             self.add_section(task)

#         report_path = self.out_dir / self.REPORT_NAME
#         doc = SimpleDocTemplate(str(report_path.resolve()), pagesize=letter)
#         doc.build(self.story)

#         print("✅ PDF created !")

#     def add_header(self):
#         title = Paragraph(
#             f'<para align="left">{self.Title_header}</para>',
#             self.STYLES["Heading1"],
#         )
#         self.story.append(title)
#         self.story.append(Spacer(1, 28))

#     def add_section(self, task):
#         matrix = self.extract_task_matrix(task)
#         table_data = self.get_table_data(task, matrix)
#         self.build_metric_section(task, table_data)

#     def extract_task_matrix(self, task):
#         deep_keys = self.DEEP_KEYS[task]
#         num_metrics = len(deep_keys)

#         data = np.full(
#             (len(EXPECTED_COILS), len(EXPECTED_ORIENTATIONS), num_metrics), np.nan
#         )

#         for i, coil in enumerate(EXPECTED_COILS):
#             for j, orientation in enumerate(EXPECTED_ORIENTATIONS):
#                 for k, metric_keys in enumerate(deep_keys):
#                     val = chained_get(
#                         self.results,
#                         task,
#                         coil,
#                         orientation,
#                         *metric_keys,
#                         default=None,
#                     )
#                     if val is not None:
#                         try:
#                             data[i, j, k] = float(val)
#                         except (TypeError, ValueError):
#                             pass

#         return data



#     def get_table_data(self, task, matrix):
#         basic_table_header = self.get_table_header(task)
#         table_data = [basic_table_header]
#         numeric_copy = []  # For conditional formatting

#         for coil_id, row_idx in self.COIL_ROW_MAP.items():
#             row_basic = self.get_basic_row_data(matrix, row_idx, coil_id)

#             if task == "Slice Thickness":
#                 diff_from_actual = abs(row_basic[-1] - self.TRUE_SLICE_THICKNESS)
#                 row_extension = [diff_from_actual]
#             elif task == "SNR":
#                 baseline = self.baselines.loc[task, coil_id]
#                 perc_diff = float(
#                     self.calc_deviation_row_mean_from_val(
#                         matrix, row_idx, baseline, representation="percentage"
#                     )
#                 )
#                 row_extension = [baseline, round(perc_diff, 2)]
#             elif task == "Geometric Accuracy":
#                 deviation = float(
#                     self.calc_deviation_row_mean_from_val(
#                         matrix, row_idx, self.TRUE_PHANTOM_DIAMETER, representation="absolute"
#                     )
#                 )
#                 deviationP = float(
#                     self.calc_deviation_row_mean_from_val(
#                         matrix, row_idx, self.TRUE_PHANTOM_DIAMETER, representation="percentage"
#                     )
#                 )
#                 row_extension = [round(deviation, 2), round(deviationP, 2)]
#             else:
#                 row_extension = []

#             full_row = row_basic + row_extension
#             table_data.append(full_row)

#             # numeric copy (skip first column)
#             numeric_copy.append([cell if isinstance(cell, numbers.Real) else None for cell in full_row])

#         return table_data, numeric_copy






#     # def get_table_data(self, task, matrix):
#     #     basic_table_header = self.get_table_header(task)
#     #     table_data = [basic_table_header]

#     #     for coil_id, row_idx in self.COIL_ROW_MAP.items():
#     #         row_basic = self.get_basic_row_data(matrix, row_idx, coil_id)

#     #         if task == "Slice Thickness":
#     #             diff_from_actual = abs(row_basic[-1] - self.TRUE_SLICE_THICKNESS)
#     #             row_extension = [diff_from_actual]

#     #         elif task == "SNR":
#     #             baseline = self.baselines.loc[task, coil_id]
#     #             perc_diff_from_baseline = round(
#     #                 float(
#     #                     self.calc_deviation_row_mean_from_val(
#     #                         matrix, row_idx, baseline, representation="percentage"
#     #                     )
#     #                 ),
#     #                 2,
#     #             )
#     #             row_extension = [baseline, perc_diff_from_baseline]

#     #         elif task == "Geometric Accuracy":
#     #             deviation = float(
#     #                 self.calc_deviation_row_mean_from_val(
#     #                     matrix,
#     #                     row_idx,
#     #                     self.TRUE_PHANTOM_DIAMETER,
#     #                     representation="absolute",
#     #                 )
#     #             )
#     #             deviationP = float(
#     #                 self.calc_deviation_row_mean_from_val(
#     #                     matrix,
#     #                     row_idx,
#     #                     self.TRUE_PHANTOM_DIAMETER,
#     #                     representation="percentage",
#     #                 )
#     #             )
#     #             row_extension = [round(deviation, 2), (deviationP/self.TRUE_PHANTOM_DIAMETER) * 100]

#     #         else:
#     #             row_extension = []

#     #         full_row = row_basic + row_extension
#     #         table_data.append(full_row)

#     #     formatted_table_data = [
#     #         [f"{cell:.2f}" if isinstance(cell, numbers.Real) else cell for cell in row]
#     #         for row in table_data
#     #     ]

#     #     return formatted_table_data

#     @staticmethod
#     def get_table_header(task):
#         HEADER_EXTENSIONS = {
#             "Slice Thickness": ["Deviation from prescribed (mm)"],
#             "SNR": ["Baseline SNR", "Deviation from baseline (%)"],
#             "Geometric Accuracy": ["Actual deviation (mm) ", "Percentage linearity error (%)"],
#             "Uniformity": [],
#             "Spatial Resolution": [],
#         }

#         basic_table_header = [" ", "Axial", "Sagittal", "Coronal", f"Average {task}"]
#         final_header = basic_table_header + HEADER_EXTENSIONS[task]
#         return final_header

#     def get_basic_row_data(self, matrix, row_idx, coil_id):
#         matrix_row = matrix[row_idx, :3]
#         if isinstance(matrix_row[0], (list, tuple, np.ndarray)):
#             matrix_row = [np.mean(x) for x in matrix_row]

#         return [
#             Paragraph(f"<b>{self.COIL_DESCRIPTIONS[coil_id]}</b>"),
#             *matrix_row,
#             np.nanmean(matrix_row),
#         ]

#     @staticmethod
#     def calc_deviation_row_mean_from_val(
#         matrix, row, reference_val, representation="absolute"
#     ):
#         row_average = np.nanmean(matrix[row, :])
#         deviation = row_average - reference_val
#         if representation == "absolute":
#             return deviation
#         elif representation == "percentage":
#             return np.abs(deviation / reference_val) * 100
#         else:
#             raise ValueError("representation must be 'absolute' or 'percentage'")
        

#     def build_metric_section(self, task, table_data_numeric_tuple):
#         table_style = deepcopy(self.TABLE_STYLE_DEFAULT)
#         table_data, numeric_copy = table_data_numeric_tuple

#         # Styles
#         wrap_style = ParagraphStyle('wrap', fontName='Helvetica', fontSize=9, leading=12)
#         wrap_style_header = ParagraphStyle(
#             'wrap_header', fontName='Helvetica-Bold', fontSize=9, leading=12, textColor=colors.white
#         )

#         # Section title
#         self.story.append(Paragraph(f"<font size=16><b>{task}</b></font>", self.STYLES["Normal"]))
#         self.story.append(Spacer(1, 20))

#         # Section subtitle
#         self.story.append(Paragraph(f"<font size=12>{self.SUBTITLES[task]}</font>", self.STYLES["Normal"]))
#         self.story.append(Spacer(1, 12))

#         # Bullet points
#         bullet_list = ListFlowable(
#             [ListItem(Paragraph(b, self.STYLES["Normal"])) for b in self.BULLET_POINTS[task]],
#             **self.BULLET_KWARGS
#         )
#         self.story.append(bullet_list)
#         self.story.append(Spacer(1, 22))

#         # Threshold conditions
#         THRESHOLD_CONDITIONS = {
#             "Slice Thickness": ">",
#             "SNR": ">",
#             "Geometric Accuracy": ">",
#             "Uniformity": "<",
#             "Spatial Resolution": ">",
#         }

#         thresholds_for_task = self.COLUMNS_TO_CHECK.get(task, [(-1, self.thresholds.get(task))])

#         # Conditional formatting
#         for row_idx in range(1, len(table_data)):
#             for col_idx, threshold in thresholds_for_task:
#                 if threshold is None:
#                     threshold = self.thresholds.get(task)
#                 if threshold is None:
#                     continue
#                 try:
#                     value = numeric_copy[row_idx-1][col_idx]
#                     if value is None:
#                         continue
#                     cell_coords = (col_idx, row_idx)
#                     if (THRESHOLD_CONDITIONS[task] == ">" and abs(value) > threshold) or \
#                     (THRESHOLD_CONDITIONS[task] == "<" and abs(value) < threshold):
#                         table_style.add("BACKGROUND", cell_coords, cell_coords, colors.salmon)
#                     else:
#                         table_style.add("BACKGROUND", cell_coords, cell_coords, colors.lightgreen)
#                 except Exception:
#                     continue

#         # Convert all cells to Paragraphs (skip double-wrapping first column)
#         display_table_data = []
#         for row_idx, row in enumerate(table_data):
#             formatted_row = []
#             for i, cell in enumerate(row):
#                 # Header row
#                 if row_idx == 0:
#                     formatted_row.append(Paragraph(str(cell), wrap_style_header))
#                 # First column is already Paragraph
#                 elif i == 0 and isinstance(cell, Paragraph):
#                     formatted_row.append(cell)
#                 # Numeric cells
#                 elif isinstance(cell, numbers.Real):
#                     formatted_row.append(Paragraph(f"{cell:.2f}", wrap_style))
#                 # Other text
#                 else:
#                     formatted_row.append(Paragraph(str(cell), wrap_style))
#             display_table_data.append(formatted_row)

#         # Table width matching text area
#         PAGE_WIDTH, PAGE_HEIGHT = letter
#         LEFT_MARGIN = RIGHT_MARGIN = 72
#         TEXT_WIDTH = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN

#         num_cols = len(display_table_data[0])
#         if num_cols == 7:
#             col_ratios = [0.25, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15]
#         else:
#             col_ratios = [1/num_cols]*num_cols
#         col_widths = [TEXT_WIDTH * r for r in col_ratios]

#         table = Table(display_table_data, colWidths=col_widths, hAlign='LEFT', repeatRows=1)
#         table.setStyle(table_style)

#         self.story.append(table)
#         self.story.append(Spacer(1, 12))

