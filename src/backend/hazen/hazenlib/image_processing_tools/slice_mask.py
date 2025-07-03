import numpy as np
import cv2
from typing import Self

from backend.hazen.hazenlib.image_processing_tools.contour_validation import (
    is_phantom_edge,
)


class SliceMask(np.ndarray):

    def __new__(cls, image, **kwargs):
        image_thresholded, contours = cls.dynamically_threshold(image, **kwargs)
        obj = np.asarray(image_thresholded).view(cls)
        obj.contours = contours
        return obj

    def __init__(self, image, **kwargs):
        self.elliptical_mask, self.centre, self.radius = self._get_elliptical_mask()

    @classmethod
    def dynamically_threshold(
        cls,
        image,
        closing_strength=7,
        mode_findContours=cv2.RETR_TREE,
        dynamic_thresh_start=4,
        additional_contour_check=None,
    ):

        image_preprocessed = cls._preprocess_image(image, closing_strength)
        dynamic_thresh = dynamic_thresh_start

        while dynamic_thresh <= 255:
            mask, contours = cls._get_mask_and_contours(
                image_preprocessed, dynamic_thresh, mode_findContours
            )

            if cls._expected_contours_visible(contours, mask, additional_contour_check):
                pad = 0
                mask_padded, contours_final = cls._get_mask_and_contours(
                    image_preprocessed, dynamic_thresh + pad, mode_findContours
                )

                mask_filtered = cls._filter_out_small_connec_comps(mask_padded)

                return mask_filtered, contours_final

            dynamic_thresh += 2

        raise ValueError("Expected phantom features not detected in mask creation!")

    @staticmethod
    def _preprocess_image(image, closing_strength):
        image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(
            "uint8"
        )

        image_denoised = cv2.fastNlMeansDenoising(
            image_normalized,
            h=np.std(image_normalized) * 0.25,
            templateWindowSize=3,
            searchWindowSize=11,
        )

        kernel_closing = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (closing_strength, closing_strength)
        )
        image_closed = cv2.morphologyEx(image_denoised, cv2.MORPH_CLOSE, kernel_closing)

        return image_closed

    @staticmethod
    def _get_mask_and_contours(image, thresh, mode_findContours):
        _, mask = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, mode_findContours, cv2.CHAIN_APPROX_SIMPLE)
        return mask, contours

    @staticmethod
    def _expected_contours_visible(contours, mask, additional_contour_check):
        phantom_edge_visible = any([is_phantom_edge(c, mask.shape) for c in contours])
        additional_contour_visible = (
            any([additional_contour_check(c) for c in contours])
            if additional_contour_check is not None
            else True
        )
        return phantom_edge_visible and additional_contour_visible

    @staticmethod
    def _filter_out_small_connec_comps(mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        min_connected_pixels = mask.size // 100

        mask_filtered = np.zeros_like(mask)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_connected_pixels:
                mask_filtered[labels == label] = 255

        return mask_filtered

    def _get_elliptical_mask(self):
        # Take the first contour that fits that criteria for the phantom edge.
        phantom_edge = [c for c in self.contours if is_phantom_edge(c, self.shape)][0]
        params = cv2.fitEllipse(phantom_edge)

        centre = params[0]
        major_radius = params[1][0] / 2
        minor_radius = params[1][1] / 2
        approximate_radius = np.sqrt(major_radius * minor_radius)

        elliptical_mask = cv2.ellipse(
            np.zeros_like(self), params, color=255, thickness=-1
        )

        return elliptical_mask, centre, approximate_radius

    def get_scaled_mask(self, scale_factor):
        new_dims = tuple([int(scale_factor * dim) for dim in self.shape])
        scaled_mask = cv2.resize(self, new_dims, interpolation=cv2.INTER_NEAREST)

        obj = np.asarray(scaled_mask).view(type(self))
        obj.contours = [
            (c.astype(np.float32) * scale_factor).astype(np.int32)
            for c in self.contours
        ]
        return obj

    def get_rotated_mask(self, theta: float | int) -> Self:
        # Calculate the rotation matrix
        M = cv2.getRotationMatrix2D(self.centre, theta, 1.0)

        # get corners coords of self and add a column of ones to make each coord homogenous (so can use with matmul)
        h, w = self.shape
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])
        corners = np.hstack([corners, np.ones((4, 1))])

        # rotate corners by rotation matrix
        M_3x3 = np.vstack([M, [0, 0, 1]])
        corners_rot = (M_3x3 @ corners.T).T

        # get x and y bounds of rotated corners
        x_min, y_min, _ = np.min(corners_rot, axis=0)
        x_max, y_max, _ = np.max(corners_rot, axis=0)

        # get new dims for rotated image
        w_new = int(np.ceil(x_max - x_min))
        h_new = int(np.ceil(y_max - y_min))

        # Adjust the rotation matrix to account for translation (shifting the image to prevent clipping)
        M[0, 2] -= x_min
        M[1, 2] -= y_min

        # Rotate the image and resize it to the new size (without clipping)
        mask_rotated = cv2.warpAffine(
            self,
            M,
            (w_new, h_new),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        contours_rotated = [
            (cnt.reshape(-1, 2) @ M[:, :2].T + M[:, 2])
            .reshape(-1, 1, 2)
            .astype(np.int32)
            for cnt in self.contours
        ]

        elliptical_mask_rotated = cv2.warpAffine(
            self.elliptical_mask,
            M,
            (w_new, h_new),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        centre_rotated = (np.array(self.centre) @ M[:, :2].T) + M[:, 2]

        def invert_affine_matrix(M):
            R = M[:, :2]
            T = M[:, 2]
            R_inv = np.linalg.inv(R)
            T_inv = -R_inv @ T
            M_inv = np.zeros_like(M)
            M_inv[:, :2] = R_inv
            M_inv[:, 2] = T_inv
            return M_inv

        obj = np.asarray(mask_rotated).view(type(self))
        obj.contours = contours_rotated
        obj.elliptical_mask = elliptical_mask_rotated
        obj.centre = centre_rotated
        obj.radius = self.radius

        M_inv = invert_affine_matrix(M)
        obj.transform_point_to_orig_frame = (
            lambda p: (np.array(p) @ M_inv[:, :2].T) + M_inv[:, 2]
        )

        return obj
