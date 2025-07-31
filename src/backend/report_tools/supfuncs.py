import numpy as np
from reportlab.platypus import Paragraph, Spacer, Table, ListFlowable, ListItem
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


def extract_measurement_matrix(
    formatted_results, measurement_type, metric_keys_list, coils, orientations
):
    """
    Extracts a 3D numpy array of measurement values when multiple metric keys are provided.

    Parameters:
        formatted_results (dict): Source data.
        measurement_type (str): e.g., 'SNR', 'Uniformity'.
        metric_keys_list (list of list of str): List of nested key paths. Each inner list is a path to a metric.
                                                Example: [['measurement', 'metrics', 'snr1'], ['measurement', 'metrics', 'snr2']]
        coils (list): Coil names (e.g., ['IB', 'HN', 'Flex']).
        orientations (list): Orientations (e.g., ['Ax', 'Sag', 'Cor']).

    Returns:
        np.ndarray: A 3D array of shape [num_coils, num_orientations, num_metrics].
    """
    num_metrics = len(metric_keys_list)
    data = np.full((len(coils), len(orientations), num_metrics), np.nan)

    for i, coil in enumerate(coils):
        for j, orientation in enumerate(orientations):
            for k, metric_keys in enumerate(metric_keys_list):
                try:
                    result = formatted_results[measurement_type][coil][orientation]
                    for key in metric_keys:
                        result = result[key]
                    data[i, j, k] = float(result)
                except (KeyError, TypeError, ValueError):
                    data[i, j, k] = np.nan  # Fallback if missing or not a number

    return data


def build_metric_section(
    story,
    title,
    subtitle,
    bullet_points,
    table_data,
    thresholds=None,
    threshold_column_index=None,
    threshold_condition=">",
    highlight_colors=(colors.salmon, colors.lightgreen),
    style=None,
):
    styles = getSampleStyleSheet()
    # Section title
    story.append(Paragraph(f"<font size=16><b>{title}</b></font>", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Section subtitle
    story.append(Paragraph(f"<font size=12>{subtitle}</font>", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Bullet list
    bullet_list = ListFlowable(
        [ListItem(Paragraph(b, styles["Normal"])) for b in bullet_points],
        bulletType="bullet",
        bulletFontName="Helvetica",
        bulletColor=colors.black,
        bulletFontSize=14,
        bulletIndent=0,
    )
    story.append(bullet_list)
    story.append(Spacer(1, 22))

    # Apply conditional formatting
    if thresholds is not None and threshold_column_index is not None:
        for row_idx in range(
            1, len(table_data)
        ):  # chooses column 4/6 or whatever is sent in, and goes through rows
            try:
                value = float(table_data[row_idx][threshold_column_index])
                cell_coords = (threshold_column_index, row_idx)

                # Comparison condition
                if threshold_condition == ">" and abs(value) > thresholds[row_idx - 1]:
                    style.add(
                        "BACKGROUND", cell_coords, cell_coords, highlight_colors[0]
                    )
                else:
                    style.add(
                        "BACKGROUND", cell_coords, cell_coords, highlight_colors[1]
                    )
            except Exception:
                continue

    # Create and style table
    table = Table(table_data)
    table.setStyle(style)
    story.append(table)
    story.append(Spacer(1, 12))
