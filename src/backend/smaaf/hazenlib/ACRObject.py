import cv2
import scipy
import skimage
import numpy as np
from pydicom import dcmread
import matplotlib.pyplot as plt
from hazenlib.utils import get_image_orientation
from pydicom.pixel_data_handlers.util import apply_modality_lut
import os


class ACRObject:
    def __init__(self, dcm_list, kwargs={}):

        # Added in a medium ACR phantom flag, not sure if this is the best way of doing this but will leave it for now..
        self.MediumACRPhantom = False
        if "MediumACRPhantom" in kwargs.keys():
            self.MediumACRPhantom = kwargs["MediumACRPhantom"]

        """
        Added in a flag to make use of the dot matrix instead of MTF for spatial res
        self.UseDotMatrix=False
        if "UseDotMatrix" in kwargs.keys():
        self.UseDotMatrix =kwargs["UseDotMatrix"]
        """

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

        # Check slice order of images and reverse if necessary. Mask of uniformity slice (slice 4) is also stored.
        self.masks[4] = self.slice_order_checks()

        # Get mask of slice thickness slice (slice 0)
        self.masks[0] = self.get_mask_slice_0()

        # Finds phantom centre
        self.centre, self.radius = self.find_phantom_center()

        if "Localiser" in kwargs.keys():
            self.LocalisierDCM = dcmread(kwargs["Localiser"])
        else:
            self.LocalisierDCM = None

        self.kwargs = kwargs

        # self.LR_orientation_checks()

        # Determine whether image rotation is necessary
        # self.rot_angle = self.determine_rotation()

        # Store the DCM object of slice 7 as it is used often
        # self.slice7_dcm = self.dcms[6]

        # Find the centre coordinates of the phantom (circle)

        # Store a mask image of slice 7 for reusability
        # self.mask_image = self.get_mask_image(self.images[6])

    def sort_images(self):
        """
        Sort a stack of images based on slice position.

        Returns
        -------
        img_stack : np.array
            A sorted stack of images, where each image is represented as a 2D numpy array.
        dcm_stack : pyd
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

    def slice_order_checks(self):
        """
        Perform orientation checks on a set of images to determine if slice order inversion is required.

        Description
        -----------
        This function analyzes the given set of images and their associated DICOM objects to determine if any
        adjustments are needed to restore the correct slice order.
        """

        def is_uniformity_slice(image):
            mask = self.get_dynamic_mask_image(image)
            canny = cv2.Canny(mask, threshold1=30, threshold2=50)
            contours, _ = cv2.findContours(
                canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = [
                c
                for c in contours
                if cv2.contourArea(cv2.convexHull(c)) >= (0.1 * canny.shape[0]) ** 2
            ]
            bool_check = all([self.is_phantom_edge(c, canny.shape) for c in contours])
            return bool_check, mask

        bool_check, mask = is_uniformity_slice(self.images[4])
        if bool_check:
            pass

        else:
            self.images.reverse()
            self.dcms.reverse()
            bool_check, mask = is_uniformity_slice(self.images[4])
            if bool_check:
                pass
            else:
                raise RuntimeError("Slice order checks failed.")

        return mask

    @classmethod
    def get_dynamic_mask_image(cls, image, additional_contour_check=None):
        """Creates a mask of the input image.
        Mask is obtained dynamically be testing for presence of phantom edge at different mask thresholds.
        Input image should be as noise-free as possible.

        Args:
            image (np.ndarray): Image to dynamically threshold. Must be 8-bit.

        Returns:
            np.ndarray: The masked image.
        """
        # denoise by fast non-local means denoising (similarity patch search)
        # h parameter calculated dynamically according to the std of image (gives indicating of noise levels)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        h = np.std(image) * 0.4

        image = cv2.fastNlMeansDenoising(image, h=h)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        dynamic_thresh = 4

        while dynamic_thresh <= 255:
            # get mask using dynamic thresholding and get contours.
            _, mask = cv2.threshold(image, dynamic_thresh, 255, cv2.THRESH_BINARY)
            canny = cv2.Canny(mask, threshold1=30, threshold2=50)
            contours, _ = cv2.findContours(
                canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

            # if any contours match expected shape of phantom edge, return mask of image.
            # returned mask has a pad in threshold for noise safety.
            phantom_edge_visible = any(
                [cls.is_phantom_edge(c, canny.shape) for c in contours]
            )

            additional_contour_visible = (
                any([additional_contour_check(c) for c in contours])
                if additional_contour_check is not None
                else True
            )

            if phantom_edge_visible and additional_contour_visible:
                pad = 15
                _, mask = cv2.threshold(
                    image, dynamic_thresh + pad, 255, cv2.THRESH_BINARY
                )

                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    mask, connectivity=8
                )
                min_connected_pixels = image.size // 100

                filtered_mask = np.zeros_like(mask)
                for label in range(1, num_labels):
                    area = stats[label, cv2.CC_STAT_AREA]
                    if area >= min_connected_pixels:
                        filtered_mask[labels == label] = 255

                return filtered_mask

            dynamic_thresh += 2

        raise ValueError("Expected phantom features not detected in mask creation!")

    @staticmethod
    def is_phantom_edge(contour, source_image_shape):
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

    def get_mask_slice_0(self):
        image_0 = self.images[0]
        mask = self.get_dynamic_mask_image(
            image_0, lambda c: self.is_slice_thickness_insert(c, image_0.shape)
        )
        return mask

    @staticmethod
    def is_slice_thickness_insert(contour, source_image_shape):
        height_image, width_image = source_image_shape
        _, (width, height), _ = cv2.minAreaRect(contour)

        if width < height:
            width, height = height, width

        width_check = 0.55 * width_image <= width <= 0.75 * width_image
        height_check = 0.02 * height_image <= height <= 0.06 * height_image

        return width_check and height_check

    def find_phantom_center(self):
        """
        Finds the phantom center and radius

        Returns
        -------
        float:
            The calculated phantom centre
        float:
            The calculated phantom radius
        """
        contours, _ = cv2.findContours(
            self.masks[4], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Take the first contour that fits that criteria for the phantom edge.
        phantom_edge = [
            c for c in contours if self.is_phantom_edge(c, self.masks[4].shape)
        ][0]
        centre, radius = cv2.minEnclosingCircle(phantom_edge)

        return centre, radius

    @staticmethod
    def circular_mask(centre, radius, dims):
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

    def measure_orthogonal_lengths(self, mask):
        """
        Compute the horizontal and vertical lengths of a mask, based on the centroid.

        Parameters:
        ----------
        mask    : ndarray of bool
            Boolean array of the image.

        Returns:
        ----------
        length_dict : dict
            A dictionary containing the following information for both horizontal and vertical line profiles:
            'Horizontal Start'      | 'Vertical Start' : tuple of int
                Horizontal/vertical starting point of the object.
            'Horizontal End'        | 'Vertical End' : tuple of int
                Horizontal/vertical ending point of the object.
            'Horizontal Extent'     | 'Vertical Extent' : ndarray of int
                Indices of the non-zero elements of the horizontal/vertical line profile.
            'Horizontal Distance'   | 'Vertical Distance' : float
                The horizontal/vertical length of the object.
        """
        dims = mask.shape
        dx, dy = self.pixel_spacing

        horizontal_start = (self.centre[1], 0)
        horizontal_end = (self.centre[1], dims[0] - 1)
        horizontal_line_profile = skimage.measure.profile_line(
            mask, horizontal_start, horizontal_end
        )
        horizontal_extent = np.nonzero(horizontal_line_profile)[0]
        horizontal_distance = (horizontal_extent[-1] - horizontal_extent[0]) * dx

        vertical_start = (0, self.centre[0])
        vertical_end = (dims[1] - 1, self.centre[0])
        vertical_line_profile = skimage.measure.profile_line(
            mask, vertical_start, vertical_end
        )
        vertical_extent = np.nonzero(vertical_line_profile)[0]
        vertical_distance = (vertical_extent[-1] - vertical_extent[0]) * dy

        length_dict = {
            "Horizontal Start": horizontal_start,
            "Horizontal End": horizontal_end,
            "Horizontal Extent": horizontal_extent,
            "Horizontal Distance": horizontal_distance,
            "Vertical Start": vertical_start,
            "Vertical End": vertical_end,
            "Vertical Extent": vertical_extent,
            "Vertical Distance": vertical_distance,
        }

        return length_dict

    @staticmethod
    def rotate_point(origin, point, angle):
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
