"""
acr_spatial_resolution.py

Paper documenting a computational approach to MTF in MR, by Delakis et al.:

https://www.researchgate.net/publication/26309398_Assessment_of_the_limiting_spatial_resolution_of_an_MRI_scanner_by_direct_analysis_of_the_edge_spread_function


This script contains the HazenTask subclass ACRSpatialResolution, which contains code relating to
calculating the spatial resolution for dcms in an ACR phantom image set. This is assessed by determining
the edge spread function across a finite region spanning the centre of the slice thickness insert. The line spread function
is then obtained, followed by MTF50. Finally, MTF50 is inverted to quantify the spatial resolution within the image.

Created by Yassine Azma (Adapted by Nathan Crossley 2025 for local RSCH purposes, 2025)
yassine.azma@rmh.nhs.uk

22/02/2023

"""

import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pydicom import Dataset

from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
from scipy.special import sici

from backend.hazen.hazenlib.HazenTask import HazenTask
from backend.hazen.hazenlib.ACRObject import ACRObject
from backend.hazen.hazenlib.masking_tools.slice_mask import SliceMask
from backend.hazen.hazenlib.masking_tools.contour_validation import (
    is_slice_thickness_insert,
)


class ACRSpatialResolution(HazenTask):
    """Subclass of HazenTask that contains code relating to calculating
    the spatial resolution of dcm images in the ACR phantom image set.
    """

    # Angle to try and superficially rotate slice thickness insert to.
    TARGET_THETA_INSERT = 3

    # Width and height of ROI used for MTF calculations.
    SIZE_ROI = 10

    def __init__(self, **kwargs):
        # Call initialiser of HazenTask
        super().__init__(**kwargs)

        # Instantiate ACRObject class using dcm list passed in within kwargs
        self.ACR_obj = ACRObject(self.dcm_list, kwargs)

    def run(self) -> dict:
        """Entry function for performing spatial resolution measurement
        on first image of the ACR phantom image set.

        Returns:
            dict: results are returned in a standardised dictionary
                structure specifying the task name, input DICOM Series
                Description + SeriesNumber + InstanceNumber, task measurement
                key-value pairs, optionally path to the generated images for
                visualisation.
        """
        # Identify relevant slice, dcm and mask
        target_slice = 0
        mtf_dcm = self.ACR_obj.dcms[target_slice]
        mask = self.ACR_obj.masks[target_slice]

        # Initialise results dictionary and add image description
        results = self.init_result_dict()
        results["file"] = self.img_desc(mtf_dcm)

        try:
            # get mtf of chosen dcm and mask
            mtf50 = self.get_mtf50(mtf_dcm, mask)

            # append results to the results dict
            results["measurement"] = {"mtf50": mtf50}

            # signal to user that the spatial resolution has been calculated for given dcm
            print(f"{self.img_desc(mtf_dcm)}: Spatial resolution calculated.")

        except Exception as e:

            # alert the user that spatial resolution could not be calculated and why
            print(
                f"Could not calculate the spatial resolution for {self.img_desc(mtf_dcm)} because of : {e}"
            )

        # only return reports if requested
        if self.report:
            results["report_image"] = self.report_files

        return results

    def get_mtf50(self, dcm: Dataset, mask: SliceMask) -> float:
        """Measure MTF50 of the provided dcm.
        Identify the slice thickness insert, place an ROI centre on the centre
        of one of its long edges, calculate the edge spread func, line spread func
        and finally MTF50.

        Args:
            dcm (Dataset): Dcm of chosen ACR slice for spatial resolution task.
            mask (SliceMask): Corresponding mask of chosen dcm.

        Returns:
            float: Measured MTF50 (spatial resolution inverted).
        """

        # get 'image of dcm
        self.image_orig = dcm.pixel_array

        # place roi and store pixel values, centre, x and y bounds and rotated image
        # (rotated to try and get insert to be approx self.TARGET_THETA_INSERT degrees off-axis)
        self.roi, self.roi_centre, self.roi_bounds, self.image_rotated = self.get_roi(
            mask
        )

        # get edge spread function from raw data
        self.esf = self.get_esf_raw()

        # get fitted version of esf, by fitting sine integral function, as per Delakis et al.
        self.esf_fitted = self.get_esf_fitted()

        # get raw line spread function from raw esf
        self.lsf = self.get_lsf(self.esf)

        # get fitted lsf from fitted esf
        self.lsf_fitted = self.get_lsf(self.esf_fitted)

        # get raw mtf func from raw lsf
        self.mtf = self.get_mtf(self.lsf)

        # get fitted mtf func from fitted lsf
        self.mtf_fitted = self.get_mtf(self.lsf_fitted)

        def simple_interpolate(y_input: float, y: list, x: list) -> float:
            """Linear interpolation function to get an output x from an input y.

            Args:
                y_input (float): input y-value to get corresponding x-value from.
                y (list): list of y-data used for interpolation.
                x (list): list of x-data used for interpolation.

            Returns:
                float: output x corresponding to input y.
            """

            # find index where y array becomes greater than input y-value
            crossing_index = np.where(y < y_input)[0][0]

            # get the x and y-values either side of the crossing index
            x1, x2 = x[crossing_index - 1], x[crossing_index]
            y1, y2 = y[crossing_index - 1], y[crossing_index]

            # perform linear interpolation to get x value
            x_output = x1 + (y_input - y1) * (x2 - x1) / (y2 - y1)

            return x_output

        # get mtf50 and mtf5 for reporting
        mtf50 = simple_interpolate(0.5, self.mtf_fitted[1], self.mtf_fitted[0])
        mtf05 = simple_interpolate(0.005, self.mtf_fitted[1], self.mtf_fitted[0])

        # report images if requested
        if self.report:
            fig, axes = plt.subplots(4, 1, figsize=(8, 16))

            # Define a padding for displaying ROI placement within original image (for context)
            DISPLAY_PAD = 20

            # work out x and y slice integers of "display frame" i.e. ROI + padding (for imshow of display frame)
            x1_roi, x2_roi, y1_roi, y2_roi = self.roi_bounds
            x1_display_frame = x1_roi - DISPLAY_PAD
            x2_display_frame = x2_roi + DISPLAY_PAD
            y1_display_frame = y1_roi - DISPLAY_PAD
            y2_display_frame = y2_roi + DISPLAY_PAD

            # work out x and y slice integeres relative to "display frame" (for plotting ROI rect)
            x1_roi_in_display = DISPLAY_PAD
            x2_roi_in_display = x2_display_frame - x1_display_frame - DISPLAY_PAD
            y1_roi_in_display = DISPLAY_PAD
            y2_roi_in_display = y2_display_frame - y1_display_frame - DISPLAY_PAD

            # show image only within display frame bounds
            axes[0].imshow(
                self.image_rotated[
                    y1_display_frame:y2_display_frame, x1_display_frame:x2_display_frame
                ],
            )

            # show ROI with a visible rect
            axes[0].vlines(
                [x1_roi_in_display, x2_roi_in_display],
                y1_roi_in_display,
                y2_roi_in_display,
            )
            axes[0].hlines(
                [y1_roi_in_display, y2_roi_in_display],
                x1_roi_in_display,
                x2_roi_in_display,
            )

            # On axis 1, plot scatter of raw esf values and also fitted esf profile
            axes[1].scatter(self.esf[0], self.esf[1], label="ESF")
            axes[1].plot(self.esf_fitted[0], self.esf_fitted[1], label="ESF fitted")
            axes[1].set_xlabel("Perpendicular distance from edge (mm)")
            axes[1].set_ylabel("Pixel value")
            axes[1].legend()

            # on axis 2, plot fitted lsf
            # axes[2].scatter(self.lsf[0], self.lsf[1], label="LSF")
            axes[2].plot(self.lsf_fitted[0], self.lsf_fitted[1], label="LSF fitted")
            axes[2].set_xlabel("Perpendicular distance from edge (mm)")
            axes[2].set_ylabel("Pixel value gradient")
            axes[2].legend()

            # on axis 3, plot fitted mtf
            # axes[3].scatter(self.mtf[0], self.mtf[1], label="MTF")
            axes[3].plot(self.mtf_fitted[0], self.mtf_fitted[1], label="MTF fitted")
            axes[3].set_xlabel("Spatial frequency (mm^-1)")
            axes[3].set_ylabel("MTF")
            axes[3].set_xlim(0, mtf05)
            axes[3].legend()

            # Work out where to save the plot.
            image_path = os.path.realpath(
                os.path.join(
                    self.report_path,
                    f"{self.img_desc(dcm)}_spatial_resolution.png",
                )
            )

            fig.tight_layout()

            # save the plot and save path to report files list
            fig.savefig(image_path, dpi=300)
            plt.close()
            self.report_files.append(image_path)

        return mtf50

    def get_roi(
        self, mask: SliceMask
    ) -> tuple[np.ndarray, np.ndarray, tuple[float], np.ndarray]:
        """Positions and extracts pixel array from ROI placed within
        the slice thickness insert, on the first slice of the ACR phantom
        image set. The slice mask is used for positioning.

        Args:
            mask (SliceMask): mask for spatial resolution task.

        Returns:
            tuple[np.ndarray, np.ndarray, tuple[float], np.ndarray]: Pixel array within
                ROI, ROI centre, x and y bounds of ROI, and rotated image (so slice thickness)
                insert is approx self.TARGET_THETA_INSERT off axis.
        """

        def get_insert_contour(mask: SliceMask) -> np.ndarray:
            """Gets the contour for the slice thickness insert from a binary mask.
            Does this by finding all contours and comparing properties of each to
            the expected properties.

            Args:
                mask (SliceMask): Binary mask to find slice thickness insert within.

            Returns:
                np.ndarray: Detected slice thickness insert.
            """

            # get all contours within mask
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # get the first contour that matches the criteria for the slice thickness mask
            insert = [c for c in contours if is_slice_thickness_insert(c, mask.shape)][
                0
            ]

            return insert

        # get slice thickness insert contour from mask
        insert = get_insert_contour(mask)

        # rotate image and mask so slice thickness insert should be self.TARGET_THETA_INSERT off axis
        image_rotated = self.rotate_rel_to_insert(self.image_orig, insert)
        mask_rotated = self.rotate_rel_to_insert(mask, insert)

        # get slice thickness insert after rotation
        insert_rotated = get_insert_contour(mask_rotated)

        # define the centre for the ROI that will be used for spatial resolution calcs.
        roi_centre = self.define_ROI_centre(insert_rotated)

        # define x and y slice indices for extracting ROI
        x1 = int(roi_centre[0] - self.SIZE_ROI // 2)
        x2 = int(roi_centre[0] + self.SIZE_ROI // 2)
        y1 = int(roi_centre[1] - self.SIZE_ROI // 2)
        y2 = int(roi_centre[1] + self.SIZE_ROI // 2)

        # extract ROI pixel array
        roi = image_rotated[y1:y2, x1:x2]

        return roi, roi_centre, (x1, x2, y1, y2), image_rotated

    @classmethod
    def rotate_rel_to_insert(cls, image: np.ndarray, insert: np.ndarray) -> np.ndarray:
        """Artificially rotate image such that slice thickness insert should
        be cls.TARGET_THETA_INSET off axis. This ensures that MTF
        measurement will be accurate (edge should be slightly angled).

        Args:
            image (np.ndarray): Input image to rotate
            insert (np.ndarray): Insert contour to base rotation off

        Returns:
            np.ndarray: Rotated image
        """
        # get minAreaRect around contour and correct angle for width-height assignment
        _, (w, h), theta = cv2.minAreaRect(insert)
        if w < h:
            theta = theta - 90

        # work out angle required to rotate image such that insert is cls.TARGET_THETA_INSERT off-axis
        theta_to_apply = theta - cls.TARGET_THETA_INSERT

        # store image height, width and centre for convenience
        (image_h, image_w) = image.shape[:2]
        center = (image_w // 2, image_h // 2)

        # Initialise rotation matrix
        matrix = cv2.getRotationMatrix2D(center, theta_to_apply, 1.0)

        # Get the new bounding box dimensions after rotation
        abs_cos = abs(matrix[0, 0])
        abs_sin = abs(matrix[0, 1])
        new_w = int(image_h * abs_sin + image_w * abs_cos)
        new_h = int(image_h * abs_cos + image_w * abs_sin)

        # Adjust the rotation matrix to account for translation (shifting the image to prevent clipping)
        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]

        # Rotate the image and resize it to the new size (without clipping)
        image_rotated = cv2.warpAffine(
            image,
            matrix,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return image_rotated

    @staticmethod
    def define_ROI_centre(insert: np.ndarray) -> np.ndarray:
        """Define centre of the ROI based on detected slice thickness insert.
        ROI positioned half way along the longest edge of the contour.

        Args:
            insert (np.ndarray): slice thicnkess insert contour.

        Returns:
            np.ndarray: Coordinates for ROI centre.
        """

        # get insert contours from minAreaRect
        corners = np.intp(cv2.boxPoints(cv2.minAreaRect(insert)))

        # choose an arbitrary test point
        test_point = corners[0]

        # get centre of ROI by selecting the 2nd furthest point away from the test point
        # and working out the midpoint between the two points
        centre = np.intp(
            np.mean(
                (
                    test_point,
                    sorted(
                        corners[1:],
                        key=lambda c: np.sqrt(np.sum((c - test_point) ** 2)),
                    )[1],
                ),
                axis=0,
            )
        )

        return centre

    def get_esf_raw(self) -> np.ndarray:
        """Get raw edge spread function using values using ROI
        pixel values, with raw perpendicular distances calculated
        relative to the ROI centre.

        Returns:
            np.ndarray: Raw edge spread function.
        """

        # get expected insert rotation in radians
        theta_rad = np.deg2rad(self.TARGET_THETA_INSERT)

        # define x and y normal magnitudes
        normal_x = -np.sin(theta_rad)
        normal_y = np.cos(theta_rad)

        # get x and y idx arrays for ROI
        y, x = np.indices(self.roi.shape)

        # calculate x and y displacement arrays, comparing coords to ROI centre
        dx = x - self.roi_centre[0]
        dy = y - self.roi_centre[1]

        # get array of perpendicular distances by dot product
        perp_distances = (dx * normal_x + dy * normal_y).flatten()

        # get pixel values corresponding to perpendicular distances by flattening array
        pix_vals = self.roi.flatten()

        # sort perp distance, pixel value pairs so perp distances are monotomically increasing
        sorted_idxs = np.argsort(perp_distances)
        perp_distances = perp_distances[sorted_idxs] * np.mean(
            self.ACR_obj.pixel_spacing
        )
        pix_vals = pix_vals[sorted_idxs]

        # define esf as combination of perp distances and pixel vals
        esf = np.array([perp_distances, pix_vals])

        return esf

    def get_esf_fitted(self) -> np.ndarray:
        """Fits an analytical function to raw esf.
        Modified sine integral function fitted, as recommended by Delakis et al.

        Returns:
            np.ndarray: Analytical fitted esf.
        """

        # define function for analytical esf (see Delakis et al.)
        def esf_func(x, c_1, c_2, alpha, m):
            return c_1 / np.pi * sici(alpha * np.pi * (x - m))[0] + c_1 / 2 + c_2

        # get best fit parameters by curve_fit with raw esf data
        popt, _ = curve_fit(
            esf_func,
            self.esf[0],
            self.esf[1],
            p0=[np.ptp(self.esf[1]), np.min(self.esf[1]), 1, np.median(self.esf[0])],
        )

        # get equally-spaced x array across raw input x-range
        x_range = np.linspace(self.esf[0][0], self.esf[0][-1], 1000)

        # construct fitted esf using x_range and applying analytical esf to it, with optimized params
        esf_fitted = np.array(
            [
                x_range,
                esf_func(x_range, *popt),
            ]
        )

        return esf_fitted

    def get_lsf(self, esf: np.ndarray) -> np.ndarray:
        """Returns a line spread function for an input
        edge spread function.

        Args:
            esf (np.ndarray): Input edge spread function

        Returns:
            np.ndarray: Output line spread function
        """

        # construct lsf from x-array of esf and gradient of esf
        return np.vstack([esf[0], np.gradient(esf[1], esf[0])])

    def get_mtf(self, lsf: np.ndarray) -> np.ndarray:
        """Gets mtf profile by taking fast fourier transform
        on input line spread function.

        Args:
            lsf (np.ndarray): Input line spread func.

        Returns:
            np.ndarray: Output mtf profile.
        """

        # get average x spacing of lsf
        dx = np.mean(np.diff(lsf[0]))

        # get mtf from absolute of fft of lsf
        mtf = np.abs(fft(lsf[1]))

        # normalize mtf so profile not dominated by DC offset
        mtf /= mtf[0]

        # get fft frequencies and store bool array defining positive freqs
        freqs = fftfreq(len(lsf[1]), dx)
        pos = freqs > 0

        # construct mtf profile by selecting positive frequencies and corresponding mtf magnitudes.
        mtf = np.vstack([freqs[pos], mtf[pos]])

        return mtf
