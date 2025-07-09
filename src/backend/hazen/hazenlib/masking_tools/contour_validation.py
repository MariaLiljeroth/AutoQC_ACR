"""
contour_validation.py

This script contains functions that return a bool based on whether an opencv contour meets the criteria expected
for specific pre-defined contours. For example, functions exist to check whether a particular contour is that of the
ACR phantom edge, or that of the ACR phantom slice thickness insert.

Written by Nathan Crossley 2025

"""

import cv2
import numpy as np


def abs_perc_diff(num_1: int | float, num_2: int | float) -> float:
    """Calculates the absolute percentage difference between
    two input numerical values.

    Args:
        num_1 (int | float): First numerical value.
        num_2 (int | float): Second numerical value.

    Returns:
        float: Percentage difference between the two values.
    """
    return abs((num_1 - num_2) / (num_2 + 1e-10))


def within_abs_tolerance(
    test_val: int | float, expected_val: int | float, tolerance: int | float
) -> bool:
    """Tests whether a test value falls within a symmetric absolute
    tolerance from an expected value.

    Args:
        test_val (int | float): Value to test whether falls within tolerance.
        expected_val (int | float): Expected value (ground truth).
        tolerance (int | float): Absolute tolerance, symmetric about ground truth.

    Returns:
        bool: True if test value falls within tolerance of expected value,
            False otherwise.
    """
    return expected_val - tolerance <= test_val <= expected_val + tolerance


def get_total_turning_angle_poly(contour: np.ndarray) -> float:
    """Calculates the "total turning angle" of a polygon
    approximation of an input contour. This is the total
    angular deviation accumulated throughout the whole
    length of the contour. Polygon approximation determined
    using Douglas Pecker algorithm.

    Args:
        contour (np.ndarray): Input contour.

    Returns:
        float: Total turning angle of contour polygon
            approximation, in degrees.
    """

    # determine polygon approximation of contour using Douglas Pecker
    epsilon = 0.005 * cv2.arcLength(contour, True)
    poly = cv2.approxPolyDP(contour, epsilon, True)

    # if polygon has more than two points, total turning angle can be calculated
    if poly.shape[0] > 2:

        # initialise dict to store angles
        angles = []

        for i in range(len(poly)):

            # get set of three consecutive points
            p1 = poly[i - 1][0]
            p2 = poly[i][0]
            p3 = poly[(i + 1) % len(poly)][0]

            # calculate the vectors between points 1-2 and 2-3
            v1 = p1 - p2
            v2 = p3 - p2

            # compute cosine of angle between vectors using dot product
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_theta = np.clip(cos_theta, -1.0, 1.0)

            # get true angle
            # 180 subtraction is to transform angle between vectors to angular deviation
            angle = abs(180 - np.degrees(np.arccos(cos_theta)))

            # append result
            angles.append(angle)

        return sum(angles)

    # otherwise, return optimal turning angle (360) so within tolerance for bool checking
    else:
        return 360


def get_smoothness_measure(contour: np.ndarray) -> float:
    """Calculates a measure of contour smoothness by taking ratio of
    perimeter of true contour to perimeter of Douglas Pecker polygon.
    A noisy contour will have a low smoothness.

    Args:
        contour (np.ndarray): Input contour

    Returns:
        float: Smoothness measure
    """

    # get perimeter of true contour
    perimeter_contour = cv2.arcLength(contour, True)

    # get Douglas Pecker polygon and its perimeter
    epsilon = cv2.arcLength(contour, True) * 0.01
    poly = cv2.approxPolyDP(contour, epsilon, True)
    perimeter_poly = cv2.arcLength(poly, True)

    # calculate measure of smoothness by ratio of perimeters.
    smoothness = perimeter_contour / (perimeter_poly + 1e-10)

    return smoothness


def is_slice_thickness_insert(contour: np.ndarray, source_image_shape: tuple) -> bool:
    """Checks whether the contour is the slice thickness insert by comparing its properties to
    the expected properties of such a contour.

    Property checks:
        height_check: Comparing the contour's approximate height to the insert's expected height.
        width_check: Comparing the contour's approximate width to the insert's expected width.
        check_total_turning_angle: Comparing the contour's total turning angle to the expected
            value (360 degrees). The total turning angle is the total angular deviation across the
            length of a Douglas Pecker polygon approximation of the contour.

    Args:
        contour (np.ndarray): A detected contour from cv2.findContours.
        source_image_shape (tuple): Shape of source image, used to scale expected contour properties with image scale

    Returns:
        bool: True if contour detected to be for the slice thickness insert, False otherwise
    """

    # Defining constants
    rows, cols = source_image_shape

    # get properties of true contour
    total_turning_angle = get_total_turning_angle_poly(contour)

    # Calculate properties of contour minAreaRect
    _, (width_bbox, height_bbox), _ = cv2.minAreaRect(contour)
    if width_bbox < height_bbox:
        width_bbox, height_bbox = height_bbox, width_bbox

    # Expected values for test criteria
    height_expected = 0.04 * rows
    width_expected = 0.65 * cols
    total_turning_angle_expected = 360

    # Defining tolerances
    tolerances = {"height": 0.5, "width": 0.15, "total_turning_angle": 0.2}

    # Tolerance checks
    check_height = abs_perc_diff(height_bbox, height_expected) <= tolerances["height"]
    check_width = abs_perc_diff(width_bbox, width_expected) <= tolerances["width"]
    check_total_turning_angle = (
        abs_perc_diff(total_turning_angle, total_turning_angle_expected)
        <= tolerances["total_turning_angle"]
    )

    return all([check_height, check_width, check_total_turning_angle])


def is_phantom_edge(contour: np.ndarray, source_image_shape: tuple) -> bool:
    """Checks whether the contour is the phantom edge by comparing its properties to
    the expected properties of such a contour.

    Property checks:
        check_centre_x: Checks that the convex hull centre-x is approximately in the centre of the image
        check_centre_y: Checks that the convex hull centre-y is approximately in the centre of the image.
        check_area: Checks that the contour area is approximately as expected.
        check_circularity: Checks that the circularity of the contour is approximately unity. Useful for
            detecting whether the contour is approximately circular and also for rejecting noisy edges that
            are not yet fully formed in the dynamic thresholding process.
        check_smoothness: Checks that the smoothness of the contour is low enough to be accepted. Smoothness
            measure determined by ratio of contour and DP polygon perimeters.

    Args:
        contour (np.ndarray): A detected contour from cv2.findContours.
        source_image_shape (tuple): Shape of source image, used to scale expected contour properties with image scale

    Returns:
        bool: True if contour detected to be phantom edge, False otherwise.
    """

    # Defining constants
    rows, cols = source_image_shape
    epsilon = 1e-10

    # Expected values for test criteria
    centre_x_expected = cols // 2
    centre_y_expected = rows // 2
    area_expected = 24000
    circularity_expected = 1
    smoothness_expected = 1

    # Defining tolerances
    tolerances = {
        "centre_x": 1 / 6,
        "centre_y": 1 / 6,
        "smoothness": 0.2,
        "area": 1 / 3,
        "circularity": 0.1,
    }

    # Measure properties of true contour
    area_contour = cv2.contourArea(contour)
    smoothness = get_smoothness_measure(contour)

    # Measure properties of convex hull
    hull = cv2.convexHull(contour)
    M_hull = cv2.moments(hull)
    perimeter_hull = cv2.arcLength(hull, True)
    centre_x_hull = M_hull["m10"] / (M_hull["m00"] + epsilon)
    centre_y_hull = M_hull["m01"] / (M_hull["m00"] + epsilon)

    # Calculate additional properties
    circularity = 4 * np.pi * area_contour / (perimeter_hull + epsilon) ** 2

    # Tolerance checks
    check_centre_x = (
        abs_perc_diff(centre_x_hull, centre_x_expected) <= tolerances["centre_x"]
    )
    check_centre_y = (
        abs_perc_diff(centre_y_hull, centre_y_expected) <= tolerances["centre_y"]
    )
    check_area = abs_perc_diff(area_contour, area_expected) <= tolerances["area"]
    check_circularity = (
        abs_perc_diff(circularity, circularity_expected) <= tolerances["circularity"]
    )
    check_smoothness = (
        abs_perc_diff(smoothness, smoothness_expected) <= tolerances["smoothness"]
    )

    return all(
        [
            check_centre_x,
            check_centre_y,
            check_area,
            check_circularity,
            check_smoothness,
        ]
    )
