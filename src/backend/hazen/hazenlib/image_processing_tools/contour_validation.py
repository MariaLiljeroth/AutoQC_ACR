"""
contour_validation.py

This script contains functions that return a bool based on whether an opencv contour meets the criteria expected
for specific pre-defined contours. For example, functions exist to check whether a particular contour is that of the
ACR phantom edge, or that of the ACR phantom slice thickness insert.

Written by Nathan Crossley 2025

"""

import cv2
import numpy as np


def is_slice_thickness_insert(contour: np.ndarray, source_image_shape: tuple) -> bool:
    """Checks whether the contour is the slice thickness insert by comparing its properties to
    the expected properties of such a contour.

    Property checks:
        height_check: Comparing the contour's approximate height to the insert's expected height.
        width_check: Comparing the contour's approximate width to the insert's expected width.
        perimeter_check: Comparing the contour's convex hull against its true perimeter.
                         The two should be equivalent for a perfect rectangle (which insert ideally is).

    Args:
        contour (np.ndarray): A detected contour from cv2.findContours.
        source_image_shape (tuple): Shape of source image, used to scale expected contour properties with image scale

    Returns:
        bool: True if contour detected to be for the slice thickness insert, False otherwise
    """

    # ideal height measured to be approx 4% of image height on ImageJ
    height_ideal = 0.04 * source_image_shape[0]

    # ideal width measured to be approx 65% of image width on ImageJ
    width_ideal = 0.65 * source_image_shape[1]

    # ideal perimeter calculated from convex hull of contour
    # (ideal in the sense that this is a simplified perimeter with no concavities)
    hull = cv2.convexHull(contour)
    perimeter_ideal = cv2.arcLength(hull, True)

    # contour height should be witin 50% of ideal height
    tolerance_height = 0.5 * height_ideal

    # contour width should be within 15% of ideal width
    tolerance_width = 0.15 * width_ideal

    # contour true perimeter should be within 10% of ideal perimeter
    tolerance_perimeter = 0.1 * perimeter_ideal

    # Get approximation for contour's true width and height from minAreaRect
    _, (width_true, height_true), _ = cv2.minAreaRect(contour)
    if width_true < height_true:
        width_true, height_true = height_true, width_true

    # Get contour true perimeter so can compare to convex hull
    perimeter_true = cv2.arcLength(contour, True)

    # bool check to see if contour height is within tolerance
    height_check = (
        height_ideal - tolerance_height
        <= height_true
        <= height_ideal + tolerance_height
    )

    # bool check to see if contour width is within tolerance
    width_check = (
        width_ideal - tolerance_width <= width_true <= width_ideal + tolerance_width
    )

    # bool check to see if perimeter is within tolerance
    perimeter_check = (
        perimeter_ideal - tolerance_perimeter
        <= perimeter_true
        <= perimeter_ideal + tolerance_perimeter
    )

    return height_check and width_check and perimeter_check


def is_phantom_edge(contour: np.ndarray, source_image_shape: tuple) -> bool:
    """Checks whether the contour is the phantom edge by comparing its properties to
    the expected properties of such a contour.

    Property checks:
        area_check: Checks that the area of a minEnclosingCircle is approximately equal to the convex hull area.
        x_check: Checks that the convex hull centre-x is approximately in the centre of the image
        y_check: Checks that the convex hull centre-y is approximately in the centre of the image.
        radius_check: Checks that the radius of a minEnclosingCircle is approximately as expected.
        perimeter_check: Checks that the convex hull perimeter is approximately equal to the minEnclosingCircle perimeter.
        circularity_check: Checks that the circularity of the convex hull is approximately unity.

    Args:
        contour (np.ndarray): A detected contour from cv2.findContours.
        source_image_shape (tuple): Shape of source image, used to scale expected contour properties with image scale

    Returns:
        bool: True if contour detected to be phantom edge, False otherwise.
    """

    # store source image dims for clarity
    rows, cols = source_image_shape

    # define an infinitesimally small constant to prevent zero division
    epsilon = 1e-10

    # Calculate area, perimeter and centre of contour convex hull
    hull = cv2.convexHull(contour)
    M_hull = cv2.moments(hull)
    area_hull = cv2.contourArea(hull)
    perimeter_hull = cv2.arcLength(hull, True)
    cx_hull = M_hull["m10"] / (M_hull["m00"] + epsilon)
    cy_hull = M_hull["m01"] / (M_hull["m00"] + epsilon)

    # Calculate radius, perimeter and area of contour's minEnclosingCircle
    _, radius_fit_circ = cv2.minEnclosingCircle(contour)
    perimeter_fit_circ = 2 * np.pi * radius_fit_circ
    area_fit_circ = np.pi * radius_fit_circ**2

    # Check that area of MinEnclosingCircle and convex hull are approx equivalent
    ratio_area = area_fit_circ / (area_hull + epsilon)
    area_check = ratio_area <= 1.1

    # Check that centre-x of convex hull is approx in centre of image
    ratio_x = cx_hull / cols
    x_check = 0.35 <= ratio_x <= 0.65

    # Check that centre-y of convex hull is approx in centre of image
    ratio_y = cy_hull / rows
    y_check = 0.35 <= ratio_y <= 0.65

    # Check that radius of minEnclosingCircle is between 1/4 and 1/2 of average image dim
    ratio_radius = radius_fit_circ / np.mean([rows, cols])
    radius_check = 1 / 4 <= ratio_radius <= 1 / 2

    # Check that convex hull and minEnclosingCircle perimeters are approx equivalent
    ratio_perimeter = perimeter_hull / (perimeter_fit_circ + epsilon)
    perimeter_check = 0.9 <= ratio_perimeter <= 1.1

    # Check that circularity of hull is approx unity
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
