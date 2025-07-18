import os
import numpy as np
import skimage
import cv2

from pydicom import dcmread, Dataset
from pydicom.pixel_data_handlers.util import apply_modality_lut

from src.backend.hazen.hazenlib.utils import get_image_orientation
from src.backend.hazen.hazenlib.masking_tools.slice_mask import SliceMask
from src.backend.hazen.hazenlib.masking_tools.contour_validation import (
    ContourValidation,
)


class ACRObject:
    def __init__(self, dcm_list, kwargs={}):
        # Added in a medium ACR phantom flag, not sure if this is the best way of doing this but will leave it for now..
        self.MediumACRPhantom = False
        if "MediumACRPhantom" in kwargs.keys():
            self.MediumACRPhantom = kwargs["MediumACRPhantom"]

        # Initialise an ACR object from a stack of images of the ACR phantom
        self.dcm_list = dcm_list

        # Load files as DICOM and their pixel arrays into 'images'
        self.images, self.dcms = self.sort_images()

        # Create empty masks to fill later where required
        self.masks = [None] * len(self.images)

        # Store the pixel spacing value from the first image (expected to be the same for all)
        if "PixelSpacing" in self.dcms[0]:
            self.pixel_spacing = self.dcms[0].PixelSpacing
        else:
            for elem in self.dcms[0].iterall():
                if elem.tag == (0x28, 0x30):
                    self.pixel_spacing = elem.value

        # Check slice order of images and reverse if necessary. Mask of slice thickness slice (slice 0) is also stored.
        self.masks[0] = self.slice_order_checks()

        # take masks of slices 4-6 for uniformity checks
        self.unif_test_idxs = [4, 5, 6]
        for idx in self.unif_test_idxs:

            # # intialise contour validation object so as to determine contour validation functions
            contour_validation = ContourValidation(self.images[idx])

            # get mask of particular slice, searching for phantom edge only
            self.masks[idx] = SliceMask(
                self.images[idx],
                contour_validation.phantom_edge_scorer,
            )

        # find most uniform slice
        self.most_uniform_slice = self.find_most_uniform_slice()

        if "Localiser" in kwargs.keys():
            self.LocalisierDCM = dcmread(kwargs["Localiser"])
        else:
            self.LocalisierDCM = None

        self.kwargs = kwargs

    def sort_images(self) -> tuple[list[np.ndarray], list[Dataset]]:
        """
        Sort a stack of images based on slice position.

        Returns
        -------
        img_stack : list[np.ndarray]
            A sorted stack of images, where each image is represented as a 2D numpy array.
        dcm_stack : list[DataSet]
            A sorted stack of dicoms
        """

        if "ImageOrientationPatient" in self.dcm_list[0]:
            ImageOrientationPatient = self.dcm_list[0].ImageOrientationPatient
        else:
            for elem in self.dcm_list[0].iterall():
                if elem.tag == (0x20, 0x37):
                    ImageOrientationPatient = elem.value

        # orientation=get_image_orientation(self.dcm_list[0].ImageOrientationPatient)
        orientation = get_image_orientation(ImageOrientationPatient)

        if "ImagePositionPatient" in self.dcm_list[0]:
            x = np.array([dcm.ImagePositionPatient[0] for dcm in self.dcm_list])
            y = np.array([dcm.ImagePositionPatient[1] for dcm in self.dcm_list])
            z = np.array([dcm.ImagePositionPatient[2] for dcm in self.dcm_list])
            if orientation == "Transverse":
                dicom_stack = [self.dcm_list[i] for i in np.argsort(z)]
            elif orientation == "Coronal":
                dicom_stack = [self.dcm_list[i] for i in np.argsort(y)]
            elif orientation == "Sagittal":
                dicom_stack = [self.dcm_list[i] for i in np.argsort(x)]

                # Rotate sagittals as required.
                for i, dcm in enumerate(dicom_stack):
                    rotated_image = skimage.transform.rotate(
                        dcm.pixel_array, -90, resize=False, preserve_range=True
                    )
                    rotated_pixel_array = rotated_image.astype(dcm.pixel_array.dtype)
                    dicom_stack[i].PixelData = rotated_pixel_array.tobytes()
                    dicom_stack[i].Rows, dicom_stack[i].Columns = rotated_image.shape

        else:
            print(
                "WARNING: Incompatible or missing image position patient tag, ordering based on filenames, this may lead to incompatible data structures!"
            )
            files = []
            for dcm in self.dcm_list:
                files.append(os.path.basename(dcm.filename))
            files.sort()

            dicom_stack = []
            for i in range(len(files)):
                for dcm in self.dcm_list:
                    if os.path.basename(dcm.filename) == files[i]:
                        dicom_stack.append(dcm)

        img_stack = [dicom.pixel_array for dicom in dicom_stack]
        img_stack = [
            apply_modality_lut(dicom.pixel_array, dicom).astype("uint16")
            for dicom in dicom_stack
        ]

        return img_stack, dicom_stack

    def slice_order_checks(self) -> SliceMask:
        """
        Perform orientation checks on a set of images to determine if slice order inversion is required.

        Description
        -----------
        This function analyzes the given set of images and their associated DICOM objects to determine if any
        adjustments are needed to restore the correct slice order. Checks are made based on detected contours
        from mask of uniformity slice (slice 5). Mask of slice 5 is returned for later use.

        Returns
        -------
        mask: SliceMask
            The mask of slice 5 (the uniformity slice).
        """

        try:
            mask_0 = self.get_mask_slice_0()
        except:
            try:
                self.images.reverse()
                self.dcms.reverse()
                mask_0 = self.get_mask_slice_0()
            except:
                raise ValueError(
                    "First or last slice not detected as slice thickness slice.\n Please check that you selected the correct subdirectories and that the images look as expected."
                )

        return mask_0

    def get_mask_slice_0(self) -> SliceMask:
        """
        Gets a mask of slice 0.

        Returns
        -------
        mask: SliceMask
            Mask of slice 0.

        """

        # get first image in stack
        image_0 = self.images[0]

        # intialise contour validation object so as to determine contour validation functions
        contour_validation = ContourValidation(image_0)

        # create slice mask, searching for both phantom edge and slice thickness insert
        # as both should be present in first slice
        mask = SliceMask(
            image_0,
            contour_validation.phantom_edge_scorer,
            contour_validation.slice_thickness_insert_scorer,
        )
        return mask

    def find_most_uniform_slice(self) -> int:
        """Finds the index of the most uniform slice in the ACR
        phantom image set. This is used for geometric accuracy,
        uniformity and SNR calculations.

        Returns:
            int: Index of most uniform slice in image set.
        """

        # get test values for quantifying presence of structures on slices
        test_vals = [self.get_structural_test_vals(idx) for idx in self.unif_test_idxs]

        # define bins to use for all three slices for shannon entropy
        min_global = min(map(min, test_vals))
        max_global = max(map(max, test_vals))
        bins = np.linspace(min_global, max_global, 64)

        # calculate shannon entropy for each slice
        entropies = [self.calculate_shannon_entropy(tvs, bins) for tvs in test_vals]

        # most uniform slice is slice with the lowest entropy (i.e. probabilities weighted due to structure presence).
        most_uniform_slice = self.unif_test_idxs[np.argmin(entropies)]

        return most_uniform_slice

    @staticmethod
    def circular_mask(centre: tuple, radius: int, dims: tuple) -> np.ndarray:
        """
        Creates a perfectly circular mask of the phantom.

        Parameters
        ----------
        centre : tuple
            The centre coordinates of the circular mask.
        radius : int
            The radius of the circular mask.
        dims   : tuple
            The dimensions of the circular mask.

        Returns
        -------
        mask : np.array
            Circular mask of the phantom.
        """
        # Define a circular logical mask

        x = np.linspace(0, dims[0] - 1, dims[0])
        y = np.linspace(0, dims[1] - 1, dims[1])

        X, Y = np.meshgrid(x, y)
        mask = (X - centre[0]) ** 2 + (Y - centre[1]) ** 2 <= radius**2

        return mask

    def get_structural_test_vals(self, slice_idx: int) -> np.ndarray:
        """Returns test values from slice for calculating measure
        of structure presence (shannon's entropy).

        Args:
            slice_idx (int): Index for slice to get test values from.

        Returns:
            np.ndarray: Test values for quantifying structure presence.
        """

        # get image and mask associated with index
        image, true_mask = self.images[slice_idx], self.masks[slice_idx]

        # remove overall intensity gradient so does not sway entropy calcs
        image_float = image.astype(np.float32)
        image_corrected = image_float - cv2.GaussianBlur(image_float, (51, 51), 0)

        # test values extracted from within circular mask.
        # mask radius less than true mask's radius to prevent edge effects from contributing to entropy.
        test_circ_mask = self.circular_mask(
            true_mask.centre, true_mask.radius * 11 / 12, true_mask.shape
        )
        test_vals = image_corrected[test_circ_mask].reshape(-1)

        return test_vals

    def calculate_shannon_entropy(self, vals: np.ndarray, bins: np.ndarray) -> float:
        """Calculates the shannon entropy from the passed test values.
        Used as a measure of structure presence to identify most uniform
        slice in ACR phantom image set.

        Args:
            vals (np.ndarray): Test values for calculation of entropy.
            bins (np.ndarray): Pixel bins for discretising test values for entropy cals.

        Returns:
            float: Shannon entropy for test values.
        """

        # get histogram using passed bins and convert to probabilities
        hist, _ = np.histogram(vals, bins=bins)
        probs = hist / hist.sum()

        # calculate shannon entropy, encluding zero-vals (as log involved)
        shannon_entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))

        return shannon_entropy

    @staticmethod
    def rotate_point(origin: tuple, point: tuple, angle: int):
        """
        Compute the horizontal and vertical lengths of a mask, based on the centroid.

        Parameters:
        ----------
        origin : tuple
            The coordinates of the point around which the rotation is performed.
        point  : tuple
            The coordinates of the point to rotate.
        angle  : int
            Angle in degrees.

        Returns:
        ----------
        x_prime : float
            A float representing the x coordinate of the desired point after being rotated around an origin.
        y_prime : float
            A float representing the y coordinate of the desired point after being rotated around an origin.
        """
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)

        x_prime = origin[0] + c * (point[0] - origin[0]) - s * (point[1] - origin[1])
        y_prime = origin[1] + s * (point[0] - origin[0]) + c * (point[1] - origin[1])
        return x_prime, y_prime

    # @staticmethod
    # def find_n_highest_peaks(data, n, height=1):
    #     """
    #     Find the indices and amplitudes of the N highest peaks within a 1D array.

    #     Parameters:
    #     ----------
    #     data    : np.array
    #         The array containing the data to perform peak extraction on.
    #     n       : int
    #         The coordinates of the point to rotate.
    #     height  : int or float
    #         The amplitude threshold for peak identification.

    #     Returns:
    #     ----------
    #     peak_locs       : np.array
    #         A numpy array containing the indices of the N highest peaks identified.
    #     peak_heights    : np.array
    #         A numpy array containing the amplitudes of the N highest peaks identified.
    #     """
    #     peaks = scipy.signal.find_peaks(data, height)
    #     pk_heights = peaks[1]["peak_heights"]
    #     pk_ind = peaks[0]

    #     peak_heights = pk_heights[
    #         (-pk_heights).argsort()[:n]
    #     ]  # find n highest peak amplitudes
    #     peak_locs = pk_ind[(-pk_heights).argsort()[:n]]  # find n highest peak locations

    #     return np.sort(peak_locs), np.sort(peak_heights)

    # def LR_orientation_checks(self):
    #     # Find center of potentially x-axis inverted image.
    #     center, radius = self.find_phantom_center(self.images[4])
    #     img2 = self.images[1]

    #     # Generate masks of phantom, and ring about phantom edge.
    #     phantomMask = self.circular_mask(center, radius * 0.98, img2.shape)
    #     tightMask = self.circular_mask(center, radius * 0.95, img2.shape)
    #     outerMask = self.circular_mask(center, radius * 1.1, img2.shape)
    #     ringMask = np.logical_xor(tightMask, outerMask)

    #     # Apply ring mask to image and dilate result.
    #     ring = np.zeros_like(img2)
    #     ring[ringMask] = img2[ringMask]
    #     dilatedRing = cv2.dilate(
    #         ring, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)), 1
    #     )
    #     dilatedRing = cv2.GaussianBlur(dilatedRing, (3, 3), 0, 0)

    #     # Combine two images.
    #     img2Mod = img2.copy()
    #     img2Mod[~phantomMask] = dilatedRing[~phantomMask]

    #     # Create binary modified image.
    #     clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(3, 3))
    #     contrastEnhanced = clahe.apply(img2Mod)
    #     _, binary = cv2.threshold(
    #         contrastEnhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    #     )

    #     # Find contour of top L shape
    #     contours, _ = cv2.findContours(
    #         binary.astype(np.uint8), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE
    #     )

    #     contours = sorted(
    #         contours,
    #         key=lambda cont: np.max(cont[:, 0, 1]),
    #         reverse=True,
    #     )

    #     topLShape = contours[-1]

    #     # Find a bbox around top L shape
    #     bbox = cv2.boundingRect(topLShape)
    #     x, y, w, h = bbox

    #     def points_from_boundingRect(x, y, w, h):
    #         points = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

    #         return points

    #     # Split bbox into two along short edge
    #     boxL = np.intp(points_from_boundingRect(x, y, w / 2, h))
    #     boxR = np.intp(points_from_boundingRect(x + w / 2, y, w / 2, h))

    #     # Find the mean pixel value within each smaller rect
    #     maskL = cv2.fillPoly(np.zeros_like(img2), [boxL], (255, 255, 255)).astype(
    #         np.uint8
    #     )
    #     maskR = cv2.fillPoly(np.zeros_like(img2), [boxR], (255, 255, 255)).astype(
    #         np.uint8
    #     )

    #     meanL = cv2.mean(img2, mask=maskL)
    #     meanR = cv2.mean(img2, mask=maskR)

    #     # LR orientation swap if required
    #     if meanL < meanR:
    #         # print("Performing LR orientation swap to restore correct view.")
    #         flipped_images = [np.fliplr(image) for image in self.images]
    #         for index, dcm in enumerate(self.dcms):
    #             dcm.PixelData = flipped_images[index].tobytes()
    #         self.images = flipped_images
    #     else:
    #         pass
    #         # print("LR orientation swap not required.")

    # def determine_rotation(self):
    #     """
    #     Determine the rotation angle of the phantom using edge detection and the Hough transform.

    #     Returns
    #     ------
    #     rot_angle : float
    #         The rotation angle in degrees.
    #     """

    #     thresh = cv2.threshold(self.images[0], 127, 255, cv2.THRESH_BINARY)[1]
    #     if (
    #         self.MediumACRPhantom is True
    #     ):  # the above thresh doesnt work for the med phantom (not sure why but maybe it shouldnt be a flat thresh anyway...)
    #         thresh = cv2.threshold(
    #             self.images[0], np.max(self.images[0]) * 0.25, 255, cv2.THRESH_BINARY
    #         )[1]

    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #     dilate = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
    #     diff = cv2.absdiff(dilate, thresh)

    #     h, theta, d = skimage.transform.hough_line(diff)
    #     _, angles, _ = skimage.transform.hough_line_peaks(h, theta, d)

    #     # angle = np.rad2deg(scipy.stats.mode(angles)[0][0]) #This needs as specific version of scipy or you get an error (drop the last [0])
    #     angle = np.rad2deg(
    #         scipy.stats.mode(angles)[0]
    #     )  # This needs as specific version of scipy or you get an error (drop the last [0])
    #     rot_angle = angle + 90 if angle < 0 else angle - 90
    #     return rot_angle

    # def rotate_images(self):
    #     """
    #     Rotate the images by a specified angle. The value range and dimensions of the image are preserved.

    #     Returns
    #     -------
    #     np.array:
    #         The rotated images.
    #     """

    #     return skimage.transform.rotate(
    #         self.images, self.rot_angle, resize=False, preserve_range=True
    #     )

    # def get_mask_image(self, image, mag_threshold=0.05, open_threshold=500):
    #     """Create a masked pixel array
    #     Mask an image by magnitude threshold before applying morphological opening to remove small unconnected
    #     features. The convex hull is calculated in order to accommodate for potential air bubbles.

    #     Args:
    #         image (_type_): _description_
    #         mag_threshold (float, optional): magnitude threshold. Defaults to 0.05.
    #         open_threshold (int, optional): open threshold. Defaults to 500.

    #     Returns:
    #         np.array:
    #             The masked image.
    #     """
    #     test_mask = self.circular_mask(
    #         self.centre, (80 // self.pixel_spacing[0]), image.shape
    #     )

    #     test_image = image * test_mask
    #     test_vals = test_image[np.nonzero(test_image)]
    #     if np.percentile(test_vals, 80) - np.percentile(test_vals, 10) > 0.9 * np.max(
    #         image
    #     ):
    #         print(
    #             "Large intensity variations detected in image. Using local thresholding!"
    #         )
    #         initial_mask = skimage.filters.threshold_sauvola(
    #             image, window_size=3, k=0.95
    #         )
    #     else:
    #         initial_mask = image > mag_threshold * np.max(image)

    #     opened_mask = skimage.morphology.area_opening(
    #         initial_mask, area_threshold=open_threshold
    #     )
    #     final_mask = skimage.morphology.convex_hull_image(opened_mask)

    #     return final_mask
