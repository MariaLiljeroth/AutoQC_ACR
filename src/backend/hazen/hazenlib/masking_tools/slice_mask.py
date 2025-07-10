"""
slice_mask.py

This script defines a custom np.ndarray subclass SliceMask, to represent the binary mask of a given slice of the ACR phantom.
The mask is obtained by iteratively testing for the presence of the phantom edge with a dynamically increasing pixel threshold.
An additional contour checking function can be passed to the class when instantiating so that the dynamic thresholding also
continues until this contour is found and its similarity score no longer is increasing. Additional mask-related utility functions
are also included in this class.

Written by Nathan Crossley 2025

"""

import numpy as np
import cv2
from typing import Self, Type, Callable
from skimage.restoration import estimate_sigma


class SliceMask(np.ndarray):
    """Subclass of np.ndarray to represent a binary mask of a given slice of the ACR phantom.
    The binary mask is obtained by dynamically testing an increasing threshold value, searching
    for the presence of expected contours.
    """

    def __new__(
        cls,
        image: np.ndarray,
        phantom_edge_scorer: Callable,
        secondary_contour_scorer: Callable = None,
    ) -> Self:
        """Creates a new object of Self, which is a subclass of np.ndarray.

        Args:
            image (np.ndarray): Image to calculate mask from.
            phantom_edge_scorer (Callable): Function that returns a score based on similarity
                between passed contour and the phantom edge.
            secondary_contour_scorer (Callable, optional): Function that returns a score based
                on the similarity between passed contour and a secondary expected contour. Defaults to None.

        Returns:
            Self: Object of self!
        """

        # get binary mask of image by dynamic thresholding and retaining contours
        mask, contours = cls.dynamically_threshold(
            image, phantom_edge_scorer, secondary_contour_scorer
        )

        # assign a SliceMask view of mask to object variable
        obj = np.asarray(mask).view(cls)

        # store detected contours as an attribute
        obj.contours = contours

        return obj

    def __init__(
        self,
        image: np.ndarray,
        phantom_edge_scorer: Callable,
        secondary_contour_scorer: Callable = None,
    ):
        """Initialises the instance of self, with callable contour
        validation functions that were used in dynamic thresholding.
        An elliptical mask is also calculated for later processing.

        Args:
            image (np.ndarray): Image that mask was calculated from.
            phantom_edge_scorer (Callable): Function that returns a score based on similarity
                between passed contour and the phantom edge.
            secondary_contour_scorer (Callable, optional): Function that returns a score based
                on the similarity between passed contour and a secondary expected contour. Defaults to None.
        """

        # store contour detection validation functions used in dynamic thresholding process
        self.phantom_edge_scorer = phantom_edge_scorer
        self.secondary_contour_scorer = secondary_contour_scorer

        # get an elliptical mask from fitting an ellipse to the phantom edge
        # store mask, centre and radius
        self.elliptical_mask, self.centre, self.radius = self._get_elliptical_mask()

    @classmethod
    def dynamically_threshold(
        cls: Type[Self],
        image: np.ndarray,
        phantom_edge_scorer: Callable,
        secondary_contour_scorer: Callable = None,
    ) -> tuple[np.ndarray]:
        """Dynamically thresholds an input image based on passed
        contour detection functions. Presence of phantom edge is used
        as a primary indication to cease thresholding. Presence of
        secondary, user defined contour is used to cease thresholding if
        provided, once the contour is optimally defined.

        Args:
            cls (Type[Self]): class of Self
            image (np.ndarray): Image used for dynamic thresholding.
            phantom_edge_scorer (Callable): Function that returns a score based on similarity
                between passed contour and the phantom edge.
            secondary_contour_scorer (Callable, optional): Function that returns a score based
                on the similarity between passed contour and a secondary expected contour. Defaults to None.

        Returns:
            tuple[np.ndarray]: Optimum mask obtaining after dynamic thresholding and associated contours.
        """

        # preprocess image before thresholding
        image_preprocessed = cls._preprocess_image(image)

        # set the dynamic threshold to low start value
        dynamic_thresh = 4

        # dynamically increase threshold whilst still within 8bit range
        while dynamic_thresh <= 255:

            # select scorer functions used to test contour presence in this first step
            # here we want to test to see if phantom edge and secondary contour are both present,
            # as this is required for successful thresholding
            scorers = [
                scorer
                for scorer in [phantom_edge_scorer, secondary_contour_scorer]
                if scorer is not None
            ]

            # get mask and contours for a particular threshold of preprocessed image
            # get similiary scores associated with phantom edge and secondary contour
            scores, contours, mask = cls._test_contour_presence(
                image_preprocessed, dynamic_thresh, scorers
            )

            # if both the phantom edge and secondary contour are present, scores both non-zero
            if all([score != 0 for score in scores]):

                # if user doesn't want to search for a secondary contour, stop optimisation here
                if len(scores) == 1:
                    return mask, contours

                # if user does want to search for a secondary contour, then keep increasing threshold
                # until further increases reduces similarity score.
                elif len(scores) == 2:

                    # store best found similarity scores, mask and associated contours
                    best_secondary_score = scores[1]
                    best_mask = mask
                    best_contours = contours

                    # define thresholding step for this final optimisation step
                    threshold_step = 2

                    while dynamic_thresh + threshold_step <= 255:

                        # increase dynamic threshold
                        dynamic_thresh += threshold_step

                        # get score for secondary contour for newly increased dynamic threshold
                        new_scores, new_contours, new_mask = cls._test_contour_presence(
                            image_preprocessed,
                            dynamic_thresh,
                            [secondary_contour_scorer],
                        )

                        # scores is a 1-element list
                        new_secondary_score = new_scores[0]

                        # If new second score improves, store it
                        plateau_tolerance = 1e-3
                        if (
                            new_secondary_score
                            > best_secondary_score + plateau_tolerance
                        ):
                            best_secondary_score = new_secondary_score
                            best_mask = new_mask
                            best_contours = new_contours

                        # score stopped increasing so return best values
                        else:
                            break

                    return best_mask, best_contours

            # increment dynamic thresh by small amount
            dynamic_thresh += 2

        # Raise an error if dynamic thresh goes outside 8bit range because required contours not detected
        raise ValueError("Expected phantom features not detected in mask creation!")

    @staticmethod
    def _preprocess_image(image: np.ndarray) -> np.ndarray:
        """Preprocess image prior to dynamic thresholding process using normalisation and
        non-local means denoising.

        Args:
            image (np.ndarray): The slice of the ACR phantom to get a binary mask of.

        Returns:
            np.ndarray: The preprocessed image.
        """

        # normalise image in 8bit range as required for denoising
        image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(
            "uint8"
        )

        # set denoised image intial value
        image_denoised = image_normalized.copy()

        # calculate initial noise level for later reference
        sigma_original = estimate_sigma(image_normalized)

        # calculate noise required to stop iteration
        sigma_thresh = sigma_original * 0.05

        # get denoised image by non local means method, testing different h-values
        # until noise low enough to continue
        k = 0.2
        while estimate_sigma(image_denoised) > sigma_thresh and k < 2:
            image_denoised = cv2.fastNlMeansDenoising(
                image_normalized,
                h=sigma_original * k,
                templateWindowSize=7,
                searchWindowSize=21,
            )
            k += 0.2

        return image_denoised

    @staticmethod
    def _test_contour_presence(
        image: np.ndarray, thresh: int, contour_scorers: list[Callable]
    ) -> tuple[list[float], np.ndarray, np.ndarray]:
        """Tests the presence of contours by evaluating any contour
        scorer functions passed by user, for each detected contour.
        Maximum similarity score returned for each function.

        Args:
            image (np.ndarray): Image to threshold for mask.
            thresh (int): Thresholding value used for mask creation.
            contour_scorers (list[Callable]): list of contour scorer functions,
                used to extract similarity scores from contours.

        Returns:
            tuple[list[float], np.ndarray, np.ndarray]: evaluated best scores for
              scorer functions, used contours and used mask.

        """

        # threshold mask using thresholding value supplied
        _, mask = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)

        # get contours of mask, simplifying with CHAIN_APPROX_SIMPLE
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # get highest scores evaluated for each contour scorer function
        scores = [max([scorer(c) for c in contours]) for scorer in contour_scorers]

        return scores, contours, mask

    def _get_elliptical_mask(self) -> tuple[np.ndarray, tuple[float, float], float]:
        """Fits an ellipse to the phantom edge, creates elliptical mask and
        returns key ellipse paramters

        Returns:
            tuple[np.ndarray, tuple[float, float], float]: Returns the elliptical mask,
                elliptical centre and approximate radius
        """
        # select the first contour that meets criteria for phantom edge
        phantom_edge_idx = np.argmax(
            [self.phantom_edge_scorer(c) for c in self.contours]
        )
        phantom_edge = self.contours[phantom_edge_idx]

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

    # @staticmethod
    # def _filter_out_small_connec_comps(mask: np.ndarray) -> np.ndarray:
    #     """Returns a filtered mask with small groups of connected pixels removed.
    #     This helps to delicately remove artificial contours from the mask (e.g. from noise).

    #     Args:
    #         mask (np.ndarray): The mask to filter.

    #     Returns:
    #         np.ndarray: The filtered mask.
    #     """

    #     # gets total number of labels, mask with pixels labelled and additional stats
    #     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    #         mask, connectivity=8
    #     )

    #     # defines threshold number of pixels in group for group to be kept
    #     min_connected_pixels = mask.size // 100

    #     # get a blank mask of correct shape
    #     mask_filtered = np.zeros_like(mask)

    #     # operate on each group at once
    #     for label in range(1, num_labels):

    #         # get area associated with each group
    #         area = stats[label, cv2.CC_STAT_AREA]

    #         # if group area above threshold, keep group, otherwise left as 0.
    #         if area >= min_connected_pixels:
    #             mask_filtered[labels == label] = 255

    #     return mask_filtered
