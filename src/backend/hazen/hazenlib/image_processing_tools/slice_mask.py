"""
slice_mask.py

This script defines a custom np.ndarray subclass SliceMask, to represent the binary mask of a given slice of the ACR phantom.
The mask is obtained by iteratively testing for the presence of the phantom edge with a dynamically increasing pixel threshold.
An additional contour checking function (returning bool) can be passed to the class when instantiating so the dynamic thresholding
continues until this contour is found, as well as the phantom edge. Additional mask-related utility functions are also included in this class.

Written by Nathan Crossley 2025

"""

import numpy as np
import cv2
from typing import Self, Type, Callable

from backend.hazen.hazenlib.image_processing_tools.contour_validation import (
    is_phantom_edge,
)


class SliceMask(np.ndarray):
    """Subclass of np.ndarray to represent a binary mask of a given slice of the ACR phantom.
    The binary mask is obtained by dynamically testing an increasing threshold value, searching
    for the presence of expected contours.
    """

    def __new__(cls, image: np.ndarray, **kwargs) -> Self:
        """Instantiates a new instance of the SliceMask class
        Dynamic thresholding occurs here so a view of the mask can be passed as the object itself.

        Args:
            image (np.ndarray): The slice of the ACR phantom to get a binary mask of.

        Returns:
            Self: An instance of SliceMask, with image data corresponding to binary mask of input image.
        """

        # get binary mask of image by dynamic thresholding and retain contours
        mask, contours = cls.dynamically_threshold(image, **kwargs)

        # assign a SliceMask view of mask to object variable
        obj = np.asarray(mask).view(cls)

        # store detected contours as an attribute
        obj.contours = contours

        return obj

    def __init__(self, image: np.ndarray, **kwargs):
        """Initialises the instance of SliceMask, configuring
        relevant instance attributes.

        Args:
            image (np.ndarray): The slice of the ACR phantom to get a binary mask of.
        """

        # get an elliptical mask from fitting an ellipse to the phantom edge
        # store mask, centre and radius
        self.elliptical_mask, self.centre, self.radius = self._get_elliptical_mask()

    @classmethod
    def dynamically_threshold(
        cls: Type[Self],
        image: np.ndarray,
        closing_strength: int = 7,
        mode_findContours: int = cv2.RETR_TREE,
        dynamic_thresh_start: int = 4,
        additional_contour_check: Callable = None,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Returns a binary mask of the input image.
        Works by testing the presence of the phantom edge (plus an optional additional contour)
        for a dynamically increasing pixel threshold.

        Args:
            cls (Type[Self]): SliceMask class.
            image (np.ndarray): The slice of the ACR phantom to get a binary mask of.
            closing_strength (int, optional): Kernel strength for morphological closing in image preprocessing. Defaults to 7.
            mode_findContours (int, optional): Specifies the mode for contour detection in cv2.findContours. Defaults to cv2.RETR_TREE.
            dynamic_thresh_start (int, optional): Specifies the pixel threshold start value. Defaults to 4.
            additional_contour_check (Callable, optional): An optional function that must return a bool based on whether a particular
                contour meets user-defined criteria. Defaults to None.

        Returns:
            tuple[np.ndarray, list[np.ndarray]]: The binary mask and associated detected contours.
        """

        # preprocess image before thresholding
        image_preprocessed = cls._preprocess_image(image, closing_strength)

        # set the dynamic threshold to the start value
        dynamic_thresh = dynamic_thresh_start

        # dynamically increase threshold whilst still within 8bit range
        while dynamic_thresh <= 255:

            # get mask and contours for the preprocessed image and particular value for dynamic threshold
            mask, contours = cls._get_mask_and_contours(
                image_preprocessed, dynamic_thresh, mode_findContours
            )

            # check whether phantom edge and additional contour are both visible at current threshold value
            if cls._expected_contours_visible(contours, mask, additional_contour_check):

                # optionally pad threshold
                pad = 0
                mask_padded, contours_final = cls._get_mask_and_contours(
                    image_preprocessed, dynamic_thresh + pad, mode_findContours
                )

                # filter out small connected components in binary mask
                mask_filtered = cls._filter_out_small_connec_comps(mask_padded)

                return mask_filtered, contours_final

            # increment dynamic thresh by small amount
            dynamic_thresh += 2

        # Raise an error if dynamic thresh goes outside 8bit range because required contours not detected
        raise ValueError("Expected phantom features not detected in mask creation!")

    @staticmethod
    def _preprocess_image(image: np.ndarray, closing_strength: int) -> np.ndarray:
        """Preprocess image prior to dynamic thresholding process using normalisation,
        non-local means denoising and morphological closing.

        Args:
            image (np.ndarray): The slice of the ACR phantom to get a binary mask of.
            closing_strength (int): Kernel strength for morphological closing.

        Returns:
            np.ndarray: The preprocessed image.
        """

        # normalise image in 8bit range as required for denoising
        image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(
            "uint8"
        )

        # denoise image with light denoising strength proportional to image standard deviation
        image_denoised = cv2.fastNlMeansDenoising(
            image_normalized,
            h=np.std(image_normalized) * 0.25,
            templateWindowSize=3,
            searchWindowSize=11,
        )

        # perform morphological closing to close gaps at phantom edge
        kernel_closing = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (closing_strength, closing_strength)
        )
        image_closed = cv2.morphologyEx(image_denoised, cv2.MORPH_CLOSE, kernel_closing)

        return image_closed

    @staticmethod
    def _get_mask_and_contours(
        image: np.ndarray, thresh: int, mode_findContours: int
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Get a binary mask using a given pixel threshold and detect contours
        of resulting mask.

        Args:
            image (np.ndarray): Image to get mask of.
            thresh (int): Pixel threshold to apply in thresholding.
            mode_findContours (int): Mode for contour detection in cv2.findContours.

        Returns:
            tuple[np.ndarray, list[np.ndarray]]: Binary mask of image and list of associated detected contours.
        """

        # threshold mask using thresholding value supplied
        _, mask = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)

        # get contours of mask, simplifying with CHAIN_APPROX_SIMPLE
        contours, _ = cv2.findContours(mask, mode_findContours, cv2.CHAIN_APPROX_SIMPLE)

        return mask, contours

    @staticmethod
    def _expected_contours_visible(
        contours: list[np.ndarray],
        mask: np.ndarray,
        additional_contour_check: Callable,
    ) -> bool:
        """Checks whether the phantom edge and (if present) the additional user-provided
        contour are both present within the list of detected contours.

        Args:
            contours (list[np.ndarray]): list of contours within which to search for expected contours.
            mask (np.ndarray): The mask that the contours where sourced from.
            additional_contour_check (Callable): An optional function that must return a bool based on
                whether a particular contour meets user-defined criteria.

        Returns:
            bool: True if all expected contours detected, False otherwise.
        """

        # checks if phantom edge is within the list of contours
        phantom_edge_visible = any([is_phantom_edge(c, mask.shape) for c in contours])

        # checks whether the user-defined additional contour is present within list of contours
        additional_contour_visible = (
            any([additional_contour_check(c) for c in contours])
            if additional_contour_check is not None
            else True
        )

        return phantom_edge_visible and additional_contour_visible

    @staticmethod
    def _filter_out_small_connec_comps(mask: np.ndarray) -> np.ndarray:
        """Returns a filtered mask with small groups of connected pixels removed.
        This helps to delicately remove artificial contours from the mask (e.g. from noise).

        Args:
            mask (np.ndarray): The mask to filter.

        Returns:
            np.ndarray: The filtered mask.
        """

        # gets total number of labels, mask with pixels labelled and additional stats
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        # defines threshold number of pixels in group for group to be kept
        min_connected_pixels = mask.size // 100

        # get a blank mask of correct shape
        mask_filtered = np.zeros_like(mask)

        # operate on each group at once
        for label in range(1, num_labels):

            # get area associated with each group
            area = stats[label, cv2.CC_STAT_AREA]

            # if group area above threshold, keep group, otherwise left as 0.
            if area >= min_connected_pixels:
                mask_filtered[labels == label] = 255

        return mask_filtered

    def _get_elliptical_mask(self) -> tuple[np.ndarray, tuple[float, float], float]:
        """Fits an ellipse to the phantom edge, creates elliptical mask and
        returns key ellipse paramters

        Returns:
            tuple[np.ndarray, tuple[float, float], float]: Returns the elliptical mask,
                elliptical centre and approximate radius
        """
        # select the first contour that meets criteria for phantom edge
        phantom_edge = [c for c in self.contours if is_phantom_edge(c, self.shape)][0]

        # fit ellipse to selected contour to get ellipse parameters
        params = cv2.fitEllipse(phantom_edge)

        # store ellipse centre
        centre = params[0]

        # calculate approximate ellipse radius
        major_radius = params[1][0] / 2
        minor_radius = params[1][1] / 2
        approximate_radius = np.sqrt(major_radius * minor_radius)

        # create elliptical mask from ellipse params
        elliptical_mask = cv2.ellipse(
            np.zeros_like(self), params, color=255, thickness=-1
        )

        return elliptical_mask, centre, approximate_radius

    def get_scaled_mask(self, scale_factor: int) -> Type[Self]:
        """Get a scaled version of self.
        Pixel array is scaled, as well as contours.

        Args:
            scale_factor (int_): Integer to scale up self by.

        Returns:
            Type[Self]: Scaled version of self.
        """

        # get new dimensions for scaled version of self
        new_dims = tuple([int(scale_factor * dim) for dim in self.shape])

        # scale self and store in variable
        scaled_mask = cv2.resize(self, new_dims, interpolation=cv2.INTER_NEAREST)

        # instantiate new instance of self baed on scaled pixel array
        obj = np.asarray(scaled_mask).view(type(self))  #

        # ensure that stored contours are scaled to be in new scaled reference frame
        obj.contours = [
            (c.astype(np.float32) * scale_factor).astype(np.int32)
            for c in self.contours
        ]

        return obj

    def get_rotated_mask(self, theta: float | int) -> Self:
        """Get a rotated version of self.

        Args:
            theta (float | int): Angle to rotate self by, in degrees.

        Returns:
            Self: Rotated version of self.
        """
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

        # Rotate contours by applying same affine matrix
        contours_rotated = [
            (cnt.reshape(-1, 2) @ M[:, :2].T + M[:, 2])
            .reshape(-1, 1, 2)
            .astype(np.int32)
            for cnt in self.contours
        ]

        # Rotate elliptical mask by applying affine matrix again
        elliptical_mask_rotated = cv2.warpAffine(
            self.elliptical_mask,
            M,
            (w_new, h_new),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # get rotated centre by applying affine matrix
        centre_rotated = (np.array(self.centre) @ M[:, :2].T) + M[:, 2]

        def invert_affine_matrix(M: np.ndarray) -> np.ndarray:
            """Defines an affine matrix that is the inverse of the original.

            Args:
                M (np.ndarray): Affine matrix to calculate inverse of.

            Returns:
                np.ndarray: Inverse of input matrix
            """
            # get rotational component
            R = M[:, :2]

            # get translational component
            T = M[:, 2]

            # invert rotational and translational components
            R_inv = np.linalg.inv(R)
            T_inv = -R_inv @ T

            # construct inverse mask
            M_inv = np.zeros_like(M)
            M_inv[:, :2] = R_inv
            M_inv[:, 2] = T_inv

            return M_inv

        # construct an instance of self with rotated variables
        obj = np.asarray(mask_rotated).view(type(self))
        obj.contours = contours_rotated
        obj.elliptical_mask = elliptical_mask_rotated
        obj.centre = centre_rotated
        obj.radius = self.radius

        # add a callable attribute to transform a given point back into the unrotated reference frame
        M_inv = invert_affine_matrix(M)
        obj.transform_point_to_orig_frame = (
            lambda p: (np.array(p) @ M_inv[:, :2].T) + M_inv[:, 2]
        )

        return obj
