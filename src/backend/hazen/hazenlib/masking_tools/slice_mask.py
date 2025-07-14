"""
slice_mask.py

This script defines a custom np.ndarray subclass SliceMask, to represent the binary mask of a given slice of the ACR phantom.
The mask is obtained by iteratively testing for the presence of the phantom edge (and optionally an additional contour) with
a dynamically increasing pixel threshold. Contour presence is scored and an optimal thresholding value is obtained, where all contours
are clearly visible, through analysing score profiles. Additional mask-related utility functions are also included in this class.

Written by Nathan Crossley 2025

"""

import numpy as np
import numbers
import cv2
from typing import Self, Type, Callable
from skimage.restoration import estimate_sigma


class SliceMask(np.ndarray):
    """Subclass of np.ndarray to represent a binary mask of a given slice of the ACR phantom.
    The binary mask is obtained by dynamically testing an increasing threshold value, searching
    for the presence of expected contours.
    """

    COARSE_SPACING = 20
    EIGHT_BIT_MIN = 0
    EIGHT_BIT_MAX = 255

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
            Self: Object of self.
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
        contour detection functions. Searches for the presence of the
        phantom edge and optionally the presence of a user defined contour.
        Contour presence is scored for both and an optimal thresholding value
        is obtained, where all contours are clearly visible, through analysing
        score profiles.


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

        # get coarse score profile for phantom edge as needed for both scenarios below
        coarse_grid_PE, scores_PE = cls._get_score_profile(
            image_preprocessed,
            phantom_edge_scorer,
            cls.EIGHT_BIT_MIN,
            cls.EIGHT_BIT_MAX,
            cls.COARSE_SPACING,
        )

        # if only phantom edge is important
        if secondary_contour_scorer is None:

            # get the maximum score for phantom edge scores
            max_score_idx = np.argmax(scores_PE)

            # finely resample scores about the maximum in the coarse profile for more accurate global maximum detection
            fine_grid_PE, refined_scores_PE = cls._finely_sample_scores(
                image_preprocessed, phantom_edge_scorer, coarse_grid_PE, max_score_idx
            )

            # get optimal thresh from max of finely sampled score profile.
            optimal_thresh = cls._get_middle_max_thresh(fine_grid_PE, refined_scores_PE)

            # get mask and contours at threshold to return
            mask, contours = cls._threshold_for_mask_and_contours(
                image_preprocessed, optimal_thresh
            )

            return mask, contours

        # if a secondary contour is also relevant
        else:

            # get coarse score profile for that secondary contour
            _, scores_secondary = cls._get_score_profile(
                image_preprocessed,
                secondary_contour_scorer,
                cls.EIGHT_BIT_MIN,
                cls.EIGHT_BIT_MAX,
                cls.COARSE_SPACING,
            )

            # get a score profile from the product of scores
            product_scores = [x * y for x, y in zip(scores_PE, scores_secondary)]

            # if all values in product score profile are zero, indicates that both contours
            # couldn't be found simultaneously at one threshold.
            if all([x for x in product_scores if x == 0]):
                raise ValueError(
                    "A common threshold does not exist where both the phantom edge and additional contour are both visible."
                )

            # get max of coarse product profile
            max_score_idx = np.argmax(product_scores)

            # finely resample phantom edge score profile about max of product profile
            fine_grid_PE, refined_scores_PE = cls._finely_sample_scores(
                image_preprocessed,
                phantom_edge_scorer,
                coarse_grid_PE,
                max_score_idx,
            )

            # finely resample secondary contour score profile about max of product profile
            _, refined_scores_secondary = cls._finely_sample_scores(
                image_preprocessed,
                secondary_contour_scorer,
                coarse_grid_PE,
                max_score_idx,
            )

            # get resampled product profile by multiplication
            refined_scores_product = [
                x * y for x, y in zip(refined_scores_PE, refined_scores_secondary)
            ]

            # get optimal thresh from max of finely sampled product profile
            optimal_thresh = cls._get_middle_max_thresh(
                fine_grid_PE, refined_scores_product
            )

            # get mask and contours at threshold for return
            mask, contours = cls._threshold_for_mask_and_contours(
                image_preprocessed, optimal_thresh
            )

            return mask, contours

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
    def _score_contour_presence(
        image: np.ndarray,
        thresh: int | list[int],
        contour_scorer: Callable,
    ) -> float | list[float]:
        """Scores presence of a particular within a mask of a passed image,
        using a particular thresholding value. If multiple thresholds are passed,
        multiple corresponding cores are returned.

        Args:
            image (np.ndarray): Passed image to create mask from.
            thresh (int | list[int]): Threshold value(s) to create mask(s) from
            contour_scorer (Callable): Callable function to score a particular contour
                on its similarity to expected contour.

        Returns:
            float | list[float]: Calculated score(s)
        """

        # Always treat thresh as list. Store bool to indicate whether single thresh value was passed.
        is_single = isinstance(thresh, numbers.Integral)
        thresh_list = [thresh] if is_single else thresh

        # for each threshold value, take mask, find contours and append score. Appended score is the highest
        # score obtained across all contours in the mask
        scores = []
        for val in thresh_list:
            mask = cv2.threshold(image, val, 255, cv2.THRESH_BINARY)[1]
            contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
            score = max((contour_scorer(c) for c in contours), default=0)
            scores.append(score)

        # Return single value if input was a single int
        if is_single:
            return scores[0]

        # else return all scores.
        else:
            return scores

    @classmethod
    def _get_score_profile(
        cls,
        image: np.ndarray,
        contour_scorer: Callable,
        pixel_value_start: int,
        pixel_value_end: int,
        grid_spacing: int = None,
    ) -> tuple[np.ndarray, list]:
        """Returns a "score profile" for a particular image, for a particular contour scorer,
        for a particular range of pixel values. A score profile is a list of calculated contour
        presence scores, for a range of user specified threshold pixel values.

        Args:
            image (np.ndarray): Image to threshold for mask.
            contour_scorer (Callable): Contour scoring function to use.
            pixel_value_start (int): Start of range of pixel value thresholds.
            pixel_value_end (int): End of range of pixel value thresholds.
            grid_spacing (int, optional): Specifies a particular step in pixel values. Defaults to None.
                If None, a pixel value spacing of 1 is assumed (no reduced sampling rate).

        Returns:
            tuple[np.ndarray, list]: Sampling pixel value grid used and calculated score profile.
        """

        # Construct grid of pixel values to test
        # If grid spacing not provided, assume to be unity.
        if grid_spacing is None:
            grid = np.arange(pixel_value_start, pixel_value_end)

        # Else use specific spacing.
        else:
            grid = np.linspace(
                pixel_value_start, pixel_value_end, grid_spacing, dtype=int
            )

        # get scores for each pixel value in grid
        score_profile = cls._score_contour_presence(image, grid, contour_scorer)

        # if all scores zero, contour not detected, so throw error
        if max(score_profile) == 0:
            raise ValueError(
                f"The contour that you are searching for with function '{contour_scorer.__name__}' does not exist on this slice!"
            )

        return grid, score_profile

    @classmethod
    def _finely_sample_scores(
        cls,
        image: np.ndarray,
        contour_scorer: Callable,
        coarse_grid: np.ndarray,
        centre_idx_grid: int,
    ) -> tuple[np.ndarray, list]:
        """Finely resampled score profile for a particular contour scorer around
        a particular idx (range is -1 to +1).

        Args:
            image (np.ndarray): The image used to create masks.
            contour_scorer (Callable): The contour scorer used to create finely sampled score profile.
            coarse_grid (np.ndarray): Grid used for course sampling (full of coarsely sampled pixel values).
            centre_idx_grid (int): Index of coarse_grid about which to construct finely sampled pixel array.

        Returns:
            tuple[np.ndarray, list]: Finely sampled grid and associated scores.
        """

        # Select start pixel value for finely sampled grid as centre index - 1 (accounting for boundary conditions)
        fine_grid_start = coarse_grid[max(centre_idx_grid - 1, 0)]

        # Select end pixel value for finely sampled grid as centre index + 1 (accounting for boundary conditions)
        fine_grid_end = coarse_grid[min(centre_idx_grid + 1, len(coarse_grid) - 1)]

        # get new finely sampled score profile
        fine_grid, refined_scores = cls._get_score_profile(
            image, contour_scorer, fine_grid_start, fine_grid_end
        )

        return fine_grid, refined_scores

    @staticmethod
    def _threshold_for_mask_and_contours(
        image: np.ndarray, thresh: int
    ) -> tuple[np.ndarray]:
        """Returns the input image, thresholded at a particular pixel value,
        and its associated contours.

        Args:
            image (np.ndarray): Image to threshold into mask.
            thresh (int): Value to use for thresholding.

        Returns:
            tuple[np.ndarray]: Binary mask and associated detected contours.
        """

        # get mask at particualr threshold value
        _, mask = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)

        # get associated contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return mask, contours

    @staticmethod
    def _get_middle_max_thresh(grid: np.ndarray, scores: list) -> float:
        """Gets the optimal thresholding value from a score profile
        by selecting the threshold associated with the middle value
        from array of maximum values.

        Args:
            grid (np.ndarray): Grid of pixel values.
            scores (list): Scores associated with pixel values.

        Returns:
            float: Optimal threshold.
        """

        # get maximum score and indices where score is maximum
        max_val = np.max(scores)
        max_indices = np.where(scores == max_val)[0]

        # extract middle value from array
        middle_max_idx = max_indices[len(max_indices) // 2]

        return grid[middle_max_idx]

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
