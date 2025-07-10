"""
contour_validation.py

This script defines a class, ContourValidation, that is used for checking calculated contours to see if they match
expected ones. Contours that can be checked for are:

The phantom edge.
The slice thickness insert.

Contour checking functions return a score based on the similarity between passed and expected contours. Scores are
calculated by combining various custom contour metrics, with different weights.

Written by Nathan Crossley 2025

"""

import cv2
import numpy as np


class ContourValidation:
    """Custom class for contour validation and checking.

    Contours can be passed into various validation functions to check
    their similarity against various expected contours. Currently supported
    contours are:

    The phantom edge.
    The slice thickness insert.

    Contour checking functions return a score based on the similarity between passed and expected contours. Scores are
    calculated by combining various custom contour metrics, with different weights. Some metrics are required and force
    return a score of 0 if not present. Other metrics are "scored" and hence contribute to the score based on percentage
    different from expected value (ground truth).
    """

    def __init__(self, source_image: np.ndarray):
        """Initialise ContourValidation class, storing image that contours
        are to be sourced from as an instance attribute, for convenience.

        Args:
            source_image (np.ndarray): Source image that contours are to be taken from.
        """

        # store source image that contours have come from for later use.
        self.source_image = source_image

    def phantom_edge_scorer(self, contour: np.ndarray) -> float:
        """Returns a float score representing the similarity of the passed
        contour to the phantom edge. Various contour properties are used with
        different weights to calculate this score.

        Property checks:
            centre_x_hull_ratio: Checks that the convex hull centre-x is approximately in the centre of the image.
            centre_y_hull_ratio: Checks that the convex hull centre-y is approximately in the centre of the image.
            area_ratio: Checks that the contour area is approximately as expected.
            circularity: Checks that the circularity of the contour is approximately unity. Useful for
                detecting whether the contour is approximately circular and also for rejecting noisy edges that
                are not yet fully formed in the dynamic thresholding process.
            smoothness_ratio: Checks that the smoothness of the contour is low enough to be accepted. Smoothness
                measure determined by ratio of contour and DP polygon perimeters.

        Args:
            contour (np.ndarray): A detected contour from cv2.findContours.

        Returns:
            float: Similarity score between passed and expected contour.
        """

        # Defining constants for convenience
        rows, cols = self.source_image.shape
        epsilon = 1e-10

        # Define dict for tests to run, defining type of test, what it is, the ground truth,
        # the tolerance to apply and the test value.
        tests_to_run = {
            "required": {
                "centre_x_hull_ratio": {
                    "ground_truth": 1 / 2,
                    "tolerance": 1 / 6,
                    "test_value": None,
                },
                "centre_y_hull_ratio": {
                    "ground_truth": 1 / 2,
                    "tolerance": 1 / 6,
                    "test_value": None,
                },
                "area_ratio": {
                    "ground_truth": 0.4,
                    "tolerance": 0.1,
                    "test_value": None,
                },
            },
            "scored": {
                "circularity": {
                    "ground_truth": 1,
                    "tolerance": 0.1,
                    "weight": 0.5,
                    "test_value": None,
                },
                "smoothness_ratio": {
                    "ground_truth": 1,
                    "tolerance": 0.2,
                    "weight": 0.5,
                    "test_value": None,
                },
            },
        }

        # Measure properties of true contour
        area_contour = cv2.contourArea(contour)
        area_ratio = area_contour / (rows * cols)
        smoothness_ratio = self.get_smoothness_ratio(contour)

        # Measure properties of convex hull
        hull = cv2.convexHull(contour)
        M_hull = cv2.moments(hull)
        perimeter_hull = cv2.arcLength(hull, True)
        centre_x_hull = M_hull["m10"] / (M_hull["m00"] + epsilon)
        centre_y_hull = M_hull["m01"] / (M_hull["m00"] + epsilon)
        centre_x_hull_ratio = centre_x_hull / cols
        centre_y_hull_ratio = centre_y_hull / rows

        # Calculate additional misc properties
        circularity = 4 * np.pi * area_contour / (perimeter_hull + epsilon) ** 2

        # append test values to dict
        tests_to_run["required"]["centre_x_hull_ratio"][
            "test_value"
        ] = centre_x_hull_ratio
        tests_to_run["required"]["centre_y_hull_ratio"][
            "test_value"
        ] = centre_y_hull_ratio
        tests_to_run["required"]["area_ratio"]["test_value"] = area_ratio
        tests_to_run["scored"]["circularity"]["test_value"] = circularity
        tests_to_run["scored"]["smoothness_ratio"]["test_value"] = smoothness_ratio

        # run tests outlined in dict to get similarity score
        score = self.run_tests(tests_to_run)

        return score

    def slice_thickness_insert_scorer(self, contour: np.ndarray) -> float:
        """Returns a float score representing the similarity of the passed
        contour to the slice thickness insert. Various contour properties
        are used with different weights to calculate this score.

        Property checks:
            width_bbox_ratio: Comparing the contour's bounding box width to the insert's expected width.
            height_bbox_ratio: Comparing the contour's bounding box height to the insert's expected height.
            smoothness_ratio: Checks that the smoothness of the contour is low enough to be accepted. Smoothness
                measure determined by ratio of contour and DP polygon perimeters.
            turning_angle_ratio: Checks that the ratio of the contour's total turning angle and 360 degrees is close
                to unity. The total turning angle is the total angular deviation across the length of a Douglas Pecker
                polygon approximation of the contour.

        Args:
            contour (np.ndarray): A detected contour from cv2.findContours.

        Returns:
            float: Similarity score between passed and expected contour.
        """

        # Defining constants for convenience
        rows, cols = self.source_image.shape

        # Define dict for tests to run, defining type of test, what it is, the ground truth,
        # the tolerance to apply and the test value.
        tests_to_run = {
            "required": {
                "width_bbox_ratio": {
                    "ground_truth": 0.65,
                    "tolerance": 0.15,
                    "test_value": None,
                },
                "height_bbox_ratio": {
                    "ground_truth": 0.04,
                    "tolerance": 0.5,
                    "test_value": None,
                },
            },
            "scored": {
                "smoothness_ratio": {
                    "ground_truth": 1,
                    "tolerance": 0.2,
                    "weight": 0.5,
                    "test_value": None,
                },
                "turning_angle_ratio": {
                    "ground_truth": 1,
                    "tolerance": 0.2,
                    "weight": 0.5,
                    "test_value": None,
                },
            },
        }

        # get properties of true contour
        turning_angle_ratio = self.get_turning_angle_ratio(contour)
        smoothness_ratio = self.get_smoothness_ratio(contour)

        # Calculate properties of contour minAreaRect
        _, (width_bbox, height_bbox), _ = cv2.minAreaRect(contour)
        if width_bbox < height_bbox:
            width_bbox, height_bbox = height_bbox, width_bbox
        width_bbox_ratio = width_bbox / cols
        height_bbox_ratio = height_bbox / rows

        # append test values to dict
        tests_to_run["required"]["width_bbox_ratio"]["test_value"] = width_bbox_ratio
        tests_to_run["required"]["height_bbox_ratio"]["test_value"] = height_bbox_ratio
        tests_to_run["scored"]["smoothness_ratio"]["test_value"] = smoothness_ratio
        tests_to_run["scored"]["turning_angle_ratio"][
            "test_value"
        ] = turning_angle_ratio

        # run tests outlined in dict to get similarity score
        score = self.run_tests(tests_to_run)

        return score

    @classmethod
    def run_tests(cls, tests_to_run: dict) -> float:
        """Runs the tests outlined in the passed dict to get a contour
        similarity score. For required tests, the score is returned as null
        if any fail. For scored tests, the tests contribute to the total score
        based on their closeness to the ground truth, assuming that they are
        below the tolerance. A quadratic relationship is implemented between score
        and percentage difference so as to especially reward close very small
        percentage differences.

        Args:
            tests_to_run (dict): Dict defining tests to run.

        Returns:
            float: Similarity score between passed and expected contours.
        """

        # first process required tests
        for test in tests_to_run["required"].values():

            # calculate percentage difference between test value and ground truth
            perc_diff = cls.abs_perc_diff(test["test_value"], test["ground_truth"])

            # if percentage difference outside of tolerance, force score to be zero
            if perc_diff >= test["tolerance"]:
                return 0.0

        # initialise a zero score and get total weights from tests
        score = 0
        total_weight = sum(test["weight"] for test in tests_to_run["scored"].values())

        # then process scored tests
        for test in tests_to_run["scored"].values():

            # calculate percentage difference between test value and ground truth
            perc_diff = cls.abs_perc_diff(test["test_value"], test["ground_truth"])

            # if percentage difference below tolerance, transform perc diff to score
            # by quadratic relationship and apply weight
            if perc_diff <= test["tolerance"]:
                score += (1 - (perc_diff / test["tolerance"]) ** 2) * test["weight"]

        # normalise score to be between 0 and 1.
        score /= total_weight

        return score

    @staticmethod
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

    @staticmethod
    def get_smoothness_ratio(contour: np.ndarray) -> float:
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

    @staticmethod
    def get_turning_angle_ratio(contour: np.ndarray) -> float:
        """Calculates the "total turning angle" of a polygon
        approximation of an input contour and its ratio to the expected
        360 degrees. The total turning angle is the total angular deviation
        accumulated throughout the wholelength of the contour. Polygon approximation
        determined using Douglas Pecker algorithm.

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

            return sum(angles) / 360

        # otherwise, return optimal turning angle ratio so within tolerance for bool checking
        else:
            return 1
