from copy import deepcopy
import numpy as np

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
)

from src.backend.report_tools.supfuncs import (
    extract_measurement_matrix,
    build_metric_section,
)

from src.shared.context import EXPECTED_COILS, EXPECTED_ORIENTATIONS


def generate_pdf_report(results, baselines, field_strength):
    ########## Setup PDF document ############
    doc = SimpleDocTemplate("AQA_Report.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    ################ Header ###################
    logo = Image(
        "src/backend/report_tools/assets/RoyalSurreyLogo.png", width=60, height=40
    )
    title = Paragraph(
        '<para align="center">Regional Radiation Protection Service</para>',
        styles["Heading1"],
    )
    header_table = Table([[title, logo]], colWidths=[450, 70])
    story.insert(0, header_table)
    story.append(Spacer(1, 22))
    paragraph_text = '<para align="center">Research & Oncology Suite, Royal Surrey County Hospital Guildford Surrey GU2 7XX Tel: 01483 408395 Email:rsc-tr.RadProt@nhs.net</para>'
    story.append(Paragraph(paragraph_text, styles["Normal"]))
    story.append(Spacer(1, 28))

    ########## Table formatting #############
    default_table_style = TableStyle(
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
    # Need to send in a fresh table style every time since cols change
    table_style_snr = deepcopy(default_table_style)
    table_style_uni = deepcopy(default_table_style)
    table_style_st = deepcopy(default_table_style)
    table_style_ga = deepcopy(default_table_style)

    ###################### Extract data from formatted_results dict ################

    # Slice Thickness
    metric_keys_list_st = [["measurement", "slice width mm"]]
    st_matrix = extract_measurement_matrix(  # This output has shape [3,3,1 or 4 depending on task] with st_matrix[0,:] = IB (ax,sag,cor), followed by HN then Flex
        results,
        measurement_type="Slice Thickness",
        metric_keys_list=metric_keys_list_st,
        coils=EXPECTED_COILS,
        orientations=EXPECTED_ORIENTATIONS,
    )

    # Uniformity
    metric_keys_list_uni = [["measurement", "integral uniformity %"]]
    uni_matrix = extract_measurement_matrix(
        results,
        measurement_type="Uniformity",
        metric_keys_list=metric_keys_list_uni,
        coils=EXPECTED_COILS,
        orientations=EXPECTED_ORIENTATIONS,
    )
    # SNR
    metric_keys_list_snr = [["measurement", "snr by smoothing", "measured"]]
    snr_matrix = extract_measurement_matrix(
        results,
        measurement_type="SNR",
        metric_keys_list=metric_keys_list_snr,
        coils=EXPECTED_COILS,
        orientations=EXPECTED_ORIENTATIONS,
    )
    snr_matrix = np.squeeze(snr_matrix)  # to get rid of empty dimensions

    # Geometric Distortion
    metric_keys_list_st = [
        ["measurement", "Horizontal distance"],
        ["measurement", "Vertical distance"],
        ["measurement", "Diagonal distance SW"],
        ["measurement", "Diagonal distance SW"],
    ]
    ga_matrix = extract_measurement_matrix(
        results,
        measurement_type="Geometric Accuracy",
        metric_keys_list=metric_keys_list_st,
        coils=EXPECTED_COILS,
        orientations=EXPECTED_ORIENTATIONS,
    )
    avg_ga_matrix = np.mean(ga_matrix, axis=2)
    print(avg_ga_matrix)

    # Threshold values
    threshold_SNR = 10  # snr %
    threshold_ST = 0.7  # mm
    threshold_uni_15T = 87.5  # %
    threshold_uni_3T = 82  # %
    threshold_ga = 2  # mm

    ############ SNR table #############

    def calc_variation(matrix, row, BL):
        var = []
        average = np.mean(matrix[row, :])
        # print(average)
        var = (np.abs(average - BL) / BL) * 100
        return var

    build_metric_section(
        story=story,
        title="Signal to Noise Ratio",
        subtitle="This measurement gives an indication as to the overall performance of the scanner. <br/> In accordance with IPEM report 112, variations of less than 10 % are considered acceptable. <br/><br/> If measurements fall outside of this tolerance, this implies either an inconsistency in acquisition parameters, uneven loading or a genuine issue with the scanner. If determined to be a genuine issue with the scanner, a decrease in SNR can be indicative of: ",
        bullet_points=[
            "Increased noise levels due to e.g interference, coil element coupling or broken coil element",
            "Decreased signal levels due to poor B0 shim, issues with transmit gain or overall degradation of the system",
        ],
        table_data=[
            [
                " ",
                "Axial",
                "Sagittal",
                "Coronal",
                "Average SNR",
                "Baseline SNR",
                "Deviation from baseline (%)",
            ],
            [
                Paragraph("<b>Inbuilt Transmit-Receive Coil</b>"),
                round(snr_matrix[0, 0], 2),
                snr_matrix[0, 1],
                snr_matrix[0, 2],
                round(np.mean(snr_matrix[0, :]), 2),
                baselines.loc["SNR", "IB"],
                round(
                    float(calc_variation(snr_matrix, 0, baselines.loc["SNR", "IB"])), 2
                ),
            ],
            [
                Paragraph("<b>Head & Neck Coil</b>"),
                snr_matrix[1, 0],
                snr_matrix[1, 1],
                snr_matrix[1, 2],
                round(np.mean(snr_matrix[1, :]), 2),
                baselines.loc["SNR", "HN"],
                round(
                    float(calc_variation(snr_matrix, 1, baselines.loc["SNR", "HN"])), 2
                ),
            ],
            [
                Paragraph("<b>Flexible Phased Array Anterior Coil</b>"),
                snr_matrix[2, 0],
                snr_matrix[2, 1],
                snr_matrix[2, 2],
                round(np.mean(snr_matrix[2, :]), 2),
                baselines.loc["SNR", "Flex"],
                round(
                    float(calc_variation(snr_matrix, 2, baselines.loc["SNR", "Flex"])),
                    2,
                ),
            ],
        ],
        thresholds=[threshold_SNR] * 3,
        threshold_column_index=6,
        threshold_condition=">",
        style=table_style_snr,
    )

    ############ Uniformity table #############
    uni_thresholds = (
        [threshold_uni_3T] * 3 if field_strength == 3 else [threshold_uni_15T] * 3
    )

    build_metric_section(
        story=story,
        title="Uniformity",
        subtitle="This measurement gives an indication as to the overall performance of the imaging gradients and the homogeneity of the main magnetic field. <br/> In accordance with the ACR Phantom Test GuidancePIU should be greater than or equal to 87.5%% for MRI systems with field strengths less than 3 Tesla. PIU should be greater than or equal to 82.0%% for MRI systems with field strength of 3 Tesla.If measurements fall outside of this tolerance, this implies an issue with: ",
        bullet_points=[
            "Poor B0 shim",
            "Poor B1 shim",
            "Coil element failure/cross-talk",
            "Uneven or incomplete loading",
        ],
        table_data=[
            [" ", "Axial", "Sagittal", "Coronal", "PIU"],
            [
                Paragraph("<b>Inbuilt Transmit-Receive Coil</b>"),
                uni_matrix[0, 0],
                uni_matrix[0, 1],
                uni_matrix[0, 2],
                round(float(np.mean(uni_matrix[0, :])), 2),
            ],
            [
                Paragraph("<b>Head & Neck Coil</b>"),
                uni_matrix[1, 0],
                uni_matrix[1, 1],
                uni_matrix[1, 2],
                round(float(np.mean(uni_matrix[1, :])), 2),
            ],
            [
                Paragraph("<b>Flexible Phased Array Anterior Coil</b>"),
                uni_matrix[2, 0],
                uni_matrix[2, 1],
                uni_matrix[2, 2],
                round(float(np.mean(uni_matrix[2, :])), 2),
            ],
        ],
        thresholds=uni_thresholds,
        threshold_column_index=4,
        threshold_condition="<",
        style=table_style_uni,
    )
    # ################ SLice Thickness ###################

    avg_st_ib = round(float(np.mean(st_matrix[0, :])))
    avg_st_hn = round(float(np.mean(st_matrix[1, :])))
    avg_st_flex = round(float(np.mean(st_matrix[2, :])))
    actual_st = 5  # mm

    build_metric_section(
        story=story,
        title="Slice Thickness",
        subtitle="This measurement gives an indication as to the overall performance of the imaging gradients and the homogeneity of the main magnetic field. <br/> In accordance with IPEM report 112, variations of less than 0.7 mm for a prescribed slice thickness of 5 mm is acceptable. If measurements fall outside of this tolerance, this implies an issue with:",
        bullet_points=["Imaging gradient non-linearity", "Poor B0 shim"],
        table_data=[
            [
                " ",
                "Axial",
                "Sagittal",
                "Coronal",
                "Average",
                "Deviation from prescribed (mm)",
            ],
            [
                Paragraph("<b>Inbuilt Transmit-Receive Coil</b>"),
                st_matrix[0, 0],
                st_matrix[0, 1],
                st_matrix[0, 2],
                avg_st_ib,
                np.abs(actual_st - avg_st_ib),
            ],
            [
                Paragraph("<b>Head & Neck Coil</b>"),
                st_matrix[1, 0],
                st_matrix[1, 1],
                st_matrix[1, 2],
                avg_st_hn,
                np.abs(actual_st - avg_st_hn),
            ],
            [
                Paragraph("<b>Flexible Phased Array Anterior Coil</b>"),
                st_matrix[2, 0],
                st_matrix[2, 1],
                st_matrix[2, 2],
                avg_st_flex,
                np.abs(actual_st - avg_st_flex),
            ],
        ],
        thresholds=[threshold_ST] * 3,
        threshold_column_index=5,
        threshold_condition=">",
        style=table_style_st,
    )

    # ################ Geometric Distortion ###################

    actual_length = 173  # mm

    def calc_diff_lengths(matrix, row, actual_length):
        var = []
        average = np.mean(matrix[row, :])
        var = np.abs(average - actual_length)
        return var

    build_metric_section(
        story=story,
        title="Geometric Accuracy",
        subtitle=":",
        bullet_points=["Imaging gradient non-linearity", "Poor B0 shim"],
        table_data=[
            [
                " ",
                "Axial",
                "Sagittal",
                "Coronal",
                "Average",
                "Deviation from actual length(mm)",
            ],
            [
                Paragraph("<b>Inbuilt Transmit-Receive Coil</b>"),
                avg_ga_matrix[0, 0],
                avg_ga_matrix[0, 1],
                avg_ga_matrix[0, 2],
                round(float(np.mean(avg_ga_matrix[0, :])), 2),
                calc_diff_lengths(avg_ga_matrix, 0, actual_length),
            ],
            [
                Paragraph("<b>Head & Neck Coil</b>"),
                avg_ga_matrix[1, 0],
                avg_ga_matrix[1, 1],
                avg_ga_matrix[1, 2],
                round(float(np.mean(avg_ga_matrix[1, :])), 2),
                calc_diff_lengths(avg_ga_matrix, 1, actual_length),
            ],
            [
                Paragraph("<b>Flexible Phased Array Anterior Coil</b>"),
                avg_ga_matrix[2, 0],
                avg_ga_matrix[2, 1],
                avg_ga_matrix[2, 2],
                round(float(np.mean(avg_ga_matrix[2, :])), 2),
                calc_diff_lengths(avg_ga_matrix, 2, actual_length),
            ],
        ],
        thresholds=[threshold_ga] * 3,
        threshold_column_index=5,
        threshold_condition=">",
        style=table_style_ga,
    )

    # # Build the PDF
    doc.build(story)

    print("✅ PDF created !")
