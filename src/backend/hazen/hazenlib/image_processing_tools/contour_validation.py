import cv2
import numpy as np


def is_slice_thickness_insert(contour: np.ndarray, source_image_shape: tuple):
    """
    Checks whether the contour is the slice thickness insert by comparing properties to
    expected properties.

    Parameters
    ----------
    contour : np.ndarray
        A detected contour from cv2.findContours.
    source_image_shape: tuple
        Shape of source image.

    Returns
    -------
    bool:
        True if contour is slice thickness insert, false otherwise.
    """
    # Define ideal height, width and perimeter
    height_ideal = 0.04 * source_image_shape[0]
    width_ideal = 0.65 * source_image_shape[1]
    hull = cv2.convexHull(contour)
    perimeter_ideal = cv2.arcLength(hull, True)

    # Define tolerances
    tolerance_height = 0.5 * height_ideal
    tolerance_width = 0.15 * width_ideal
    tolerance_perimeter = 0.1 * perimeter_ideal

    # Get rect true width and height from minAreaRect
    _, (width_true, height_true), _ = cv2.minAreaRect(contour)
    if width_true < height_true:
        width_true, height_true = height_true, width_true

    # Get contour true perimeter
    perimeter_true = cv2.arcLength(contour, True)

    # bool checks
    height_check = (
        height_ideal - tolerance_height
        <= height_true
        <= height_ideal + tolerance_height
    )
    width_check = (
        width_ideal - tolerance_width <= width_true <= width_ideal + tolerance_width
    )
    perimeter_check = (
        perimeter_ideal - tolerance_perimeter
        <= perimeter_true
        <= perimeter_ideal + tolerance_perimeter
    )

    return height_check and width_check and perimeter_check


def is_phantom_edge(contour, source_image_shape):
    """
    Checks whether the contour is the phantom edge by comparing properties to
    expected properties.

    Parameters
    ----------
    contour : np.ndarray
        A detected contour from cv2.findContours.
    source_image_shape: tuple
        Shape of source image.

    Returns
    -------
    bool:
        True if contour is phantom edge, false otherwise.
    """
    # Calculate key raw contour stats
    rows, cols = source_image_shape
    epsilon = 1e-10

    # Calculate key convex hull stats
    hull = cv2.convexHull(contour)
    M_hull = cv2.moments(hull)
    area_hull = cv2.contourArea(hull)
    perimeter_hull = cv2.arcLength(hull, True)
    cx_hull = M_hull["m10"] / (M_hull["m00"] + epsilon)
    cy_hull = M_hull["m01"] / (M_hull["m00"] + epsilon)

    # Calculate key min enclosing circle stats
    _, radius_fit_circ = cv2.minEnclosingCircle(contour)
    perimeter_fit_circ = 2 * np.pi * radius_fit_circ
    area_fit_circ = np.pi * radius_fit_circ**2

    # Check stats against expected
    ratio_area = area_fit_circ / (area_hull + epsilon)
    area_check = ratio_area <= 1.1

    ratio_x = cx_hull / cols
    x_check = 0.45 <= ratio_x <= 0.65

    ratio_y = cy_hull / rows
    y_check = 0.45 <= ratio_y <= 0.65

    ratio_radius = radius_fit_circ / np.mean([rows, cols])
    radius_check = 1 / 4 <= ratio_radius <= 1 / 2

    ratio_perimeter = perimeter_hull / (perimeter_fit_circ + epsilon)
    perimeter_check = 0.9 <= ratio_perimeter <= 1.1

    circularity = 4 * np.pi * area_hull / (perimeter_hull + epsilon) ** 2
    circularity_check = 0.9 <= circularity <= 1.1

    return all(
        [
            area_check,
            x_check,
            y_check,
            radius_check,
            perimeter_check,
            circularity_check,
        ]
    )
