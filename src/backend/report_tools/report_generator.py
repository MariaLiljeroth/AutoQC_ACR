from copy import deepcopy
import numbers
import sys
import numpy as np
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    ListFlowable,
    ListItem,
)

from src.backend.utils import chained_get
from src.shared.context import EXPECTED_COILS, EXPECTED_ORIENTATIONS


class ReportGenerator:

    REPORT_NAME = "AQA_Report.pdf"
    STYLES = getSampleStyleSheet()

    if getattr(sys, "frozen", False):

        LOGO_PATH = str(
            (
                Path(sys.executable).parent / "_internal/assets/royal_surrey_logo.png"
            ).resolve()
        )
    else:
        LOGO_PATH = "src/backend/assets/royal_surrey_logo.png"

    DEPARTMENT_NAME = "Regional Radiation Protection Service"
    DEPARTMENT_INFO = "Research & Oncology Suite, Royal Surrey County Hospital Guildford Surrey GU2 7XX Tel: 01483 408395 Email:rsc-tr.RadProt@nhs.net"

    TABLE_STYLE_DEFAULT = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),  # Header background
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),  # Header text color
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),  # Center alignment
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),  # Header font
            ("FONTSIZE", (0, 0), (-1, 0), 9),  # Font size
            ("BOTTOMPADDING", (0, 0), (-1, 0), 9),  # Header padding
            ("BACKGROUND", (0, 1), (-1, -1), colors.white),  # Body background
            ("GRID", (0, 0), (-1, -1), 1, colors.black),  # Borders
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
        "Geometric Accuracy": "PLACEHOLDER SUBTITLE",
        "Uniformity": "This measurement gives an indication as to the overall performance of the imaging gradients and the homogeneity of the main magnetic field. <br/> In accordance with the ACR Phantom Test Guidance PIU should be greater than or equal to 87.5% for MRI systems with field strengths less than 3 Tesla. PIU should be greater than or equal to 82.0% for MRI systems with field strength of 3 Tesla.If measurements fall outside of this tolerance, this implies an issue with: ",
        "Spatial Resolution": "PLACEHOLDER SUBTITLE",
    }

    BULLET_POINTS = {
        "Slice Thickness": ["Imaging gradient non-linearity", "Poor B0 shim"],
        "SNR": [
            "Increased noise levels due to e.g interference, coil element coupling or broken coil element",
            "Decreased signal levels due to poor B0 shim, issues with transmit gain or overall degradation of the system",
        ],
        "Geometric Accuracy": ["Imaging gradient non-linearity", "Poor B0 shim"],
        "Uniformity": [
            "Poor B0 shim",
            "Poor B1 shim",
            "Coil element failure/cross-talk",
            "Uneven or incomplete loading",
        ],
        "Spatial Resolution": [
            "PLACEHOLDER BULLET POINT",
            "PLACEHOLDER BULLET POINT",
            "PLACEHOLDER BULLET POINT",
        ],
    }

    COIL_DESCRIPTIONS = {
        "IB": "Inbuilt Transmit-Receive Coil",
        "HN": "Head & Neck Coil",
        "Flex": "Flexible Phased Array Anterior Coil",
    }

    COIL_ROW_MAP = {"IB": 0, "HN": 1, "Flex": 2}

    TRUE_SLICE_THICKNESS = 5  # mm
    TRUE_PHANTOM_DIAMETER = 173  # mm

    def __init__(self, results, baselines, field_strength, out_dir):
        self.results = results
        self.baselines = baselines
        self.field_strength = field_strength
        self.out_dir = out_dir
        self.story = []

        self.thresholds = {
            "Slice Thickness": 0.7,  # mm
            "SNR": 10,  # %
            "Geometric Accuracy": 2,  # mm:
            "Uniformity": 82 if field_strength == 3 else 87.5,  # %
            "Spatial Resolution": None,
        }

    def run(self):
        self.add_header()
        for task in self.results:
            self.add_section(task)

        report_path = self.out_dir / self.REPORT_NAME
        doc = SimpleDocTemplate(str(report_path.resolve()), pagesize=letter)
        doc.build(self.story)

        print("✅ PDF created !")

    def add_header(self):
        # construct table containing title and logo - add to story
        logo = Image(self.LOGO_PATH, 60, 40)
        title = Paragraph(
            f'<para align="center">{self.DEPARTMENT_NAME}</para>',
            self.STYLES["Heading1"],
        )
        title_table = Table([[title, logo]], colWidths=[450, 70])
        self.story.insert(0, title_table)
        self.story.append(Spacer(1, 22))

        # add paragraph for department info
        department_info = Paragraph(
            f'<para align="center">{self.DEPARTMENT_INFO}</para>', self.STYLES["Normal"]
        )
        self.story.append(department_info)
        self.story.append(Spacer(1, 28))

    def add_section(self, task):
        matrix = self.extract_task_matrix(task)
        table_data = self.get_table_data(task, matrix)
        self.build_metric_section(task, table_data)

    def extract_task_matrix(self, task):
        deep_keys = self.DEEP_KEYS[task]
        num_metrics = len(deep_keys)

        data = np.full(
            (len(EXPECTED_COILS), len(EXPECTED_ORIENTATIONS), num_metrics), np.nan
        )

        for i, coil in enumerate(EXPECTED_COILS):
            for j, orientation in enumerate(EXPECTED_ORIENTATIONS):
                for k, metric_keys in enumerate(deep_keys):
                    # Use chained_get to safely traverse: measurement_type -> coil -> orientation -> *metric_keys
                    val = chained_get(
                        self.results,
                        task,
                        coil,
                        orientation,
                        *metric_keys,
                        default=None,
                    )
                    if val is not None:
                        try:
                            data[i, j, k] = float(val)
                        except (TypeError, ValueError):
                            pass  # leave as NaN if conversion fails

        return data

    def get_table_data(self, task, matrix):
        basic_table_header = self.get_table_header(task)
        table_data = [basic_table_header]

        for coil_id, row_idx in self.COIL_ROW_MAP.items():
            row_basic = self.get_basic_row_data(matrix, row_idx, coil_id)

            if task == "Slice Thickness":
                diff_from_actual = abs(row_basic[-1] - self.TRUE_SLICE_THICKNESS)
                row_extension = [diff_from_actual]

            elif task == "SNR":
                baseline = self.baselines.loc[task, coil_id]
                perc_diff_from_baseline = round(
                    float(
                        self.calc_deviation_row_mean_from_val(
                            matrix, row_idx, baseline, representation="percentage"
                        )
                    ),
                    2,
                )
                row_extension = [baseline, perc_diff_from_baseline]

            elif task == "Geometric Accuracy":
                deviation = round(
                    float(
                        self.calc_deviation_row_mean_from_val(
                            matrix,
                            row_idx,
                            self.TRUE_PHANTOM_DIAMETER,
                            representation="absolute",
                        )
                    )
                )
                row_extension = [deviation]

            elif task == "Uniformity":
                row_extension = []

            elif task == "Spatial Resolution":
                row_extension = []

            full_row = row_basic + row_extension
            table_data.append(full_row)

        formatted_table_data = [
            [f"{cell:.2f}" if isinstance(cell, numbers.Real) else cell for cell in row]
            for row in table_data
        ]

        return formatted_table_data

    @staticmethod
    def get_table_header(task):
        HEADER_EXTENSIONS = {
            "Slice Thickness": ["Deviation from prescribed (mm)"],
            "SNR": ["Baseline SNR", "Deviation from baseline (%)"],
            "Geometric Accuracy": ["Deviation from actual length"],
            "Uniformity": [],
            "Spatial Resolution": [],
        }

        basic_table_header = [" ", "Axial", "Sagittal", "Coronal", f"Average {task}"]

        final_header = basic_table_header + HEADER_EXTENSIONS[task]

        return final_header

    def get_basic_row_data(self, matrix, row_idx, coil_id):
        matrix_row = matrix[row_idx, :3]

        # if still an iterable, average (for Geometric Accuracy) - would be better to adjust raw hazen output but don't have time.
        if isinstance(matrix_row[0], (list, tuple, np.ndarray)):
            matrix_row = [np.mean(x) for x in matrix_row]

        basic_row_data = [
            Paragraph(f"<b>{self.COIL_DESCRIPTIONS[coil_id]}</b>"),
            *matrix_row,
            np.nanmean(matrix_row),
        ]

        return basic_row_data

    @staticmethod
    def calc_deviation_row_mean_from_val(
        matrix, row, reference_val, representation="absolute"
    ):
        row_average = np.nanmean(matrix[row, :])
        deviation = row_average - reference_val
        if representation == "absolute":
            deviation = abs(deviation)
        elif representation == "percentage":
            deviation = np.abs(deviation / reference_val) * 100
        else:
            raise ValueError(
                f"representation arg expected to be 'absolute' or 'percentage' but received {representation}"
            )
        return deviation

    def build_metric_section(self, task, table_data):

        # Take deep copy of table style
        table_style = deepcopy(self.TABLE_STYLE_DEFAULT)

        # Section title
        self.story.append(
            Paragraph(f"<font size=16><b>{task}</b></font>", self.STYLES["Normal"])
        )
        self.story.append(Spacer(1, 12))

        # Section subtitle
        self.story.append(
            Paragraph(
                f"<font size=12>{self.SUBTITLES[task]}</font>", self.STYLES["Normal"]
            )
        )
        self.story.append(Spacer(1, 12))

        # Bullet list
        bullet_list = ListFlowable(
            [
                ListItem(Paragraph(b, self.STYLES["Normal"]))
                for b in self.BULLET_POINTS[task]
            ],
            **self.BULLET_KWARGS,
        )
        self.story.append(bullet_list)
        self.story.append(Spacer(1, 22))

        # Apply conditional formatting
        THRESHOLD_CONDITIONS = {
            "Slice Thickness": ">",
            "SNR": ">",
            "Geometric Accuracy": ">",
            "Uniformity": "<",
            "Spatial Resolution": ">",
        }
        threshold = self.thresholds[task]
        if threshold is not None:
            for row_idx in range(
                1, len(table_data)
            ):  # chooses column 4/6 or whatever is sent in, and goes through rows
                try:
                    value = float(table_data[row_idx][-1])
                    cell_coords = (-1, row_idx)

                    # Comparison condition
                    if (
                        THRESHOLD_CONDITIONS[task] == ">" and abs(value) > threshold
                    ) or (THRESHOLD_CONDITIONS[task] == "<" and abs(value) < threshold):
                        table_style.add(
                            "BACKGROUND", cell_coords, cell_coords, colors.salmon
                        )
                    else:
                        table_style.add(
                            "BACKGROUND", cell_coords, cell_coords, colors.lightgreen
                        )
                except Exception:
                    continue

        # Create and style table
        table = Table(table_data)
        table.setStyle(table_style)
        self.story.append(table)
        self.story.append(Spacer(1, 12))
