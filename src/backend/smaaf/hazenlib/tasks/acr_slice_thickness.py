"""
ACR Slice Thickness

Calculates the slice thickness for slice 1 of the ACR phantom.

The ramps located in the middle of the phantom are located and line profiles are drawn through them. The full-width
half-maximum (FWHM) of each ramp is determined to be their length. Using the formula described in the ACR guidance, the
slice thickness is then calculated.

Created by Yassine Azma
yassine.azma@rmh.nhs.uk

31/01/2022
"""

import os
from typing import Callable

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pydicom.dataset import FileDataset

from scipy.signal import find_peaks, medfilt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

from backend.smaaf.hazenlib.HazenTask import HazenTask
from backend.smaaf.hazenlib.ACRObject import ACRObject
from backend.smaaf.hazenlib.image_processing_tools.slice_mask import SliceMask
from backend.smaaf.hazenlib.image_processing_tools.contour_validation import (
    is_slice_thickness_insert,
)
from backend.smaaf.hazenlib.utils import Point, Line, XY


class ACRSliceThickness(HazenTask):
    """Slice width measurement class for DICOM images of the ACR phantom."""

    class SignalLine(Line):
        """Subclass of Line to implement functionality related to the ACR phantom's ramps

        Instance attributes:
            FWHM (float): Calculated FWHM value for object.
            fitted (XY): Fitted profile from sigmoid fitting model.
        """

        def get_FWHM(self):
            """Calculates the FWHM by fitting a piecewise sigmoid to signal across line"""

            # Raises error if function called without signal attribute existing.
            if not hasattr(self, "signal"):
                raise ValueError("Signal across line has not been computed!")

            # get fitted signal
            fitted = self._fit_piecewise_sigmoid()

            # find peaks of sigmoid model to extract max peak height.
            _, props = find_peaks(
                fitted.y, height=0, prominence=np.ptp(fitted.y).item() / 4
            )
            peak_height = np.max(props["peak_heights"])

            # get background pixel values on left and right.

            backgroundL = fitted.y[0]
            backgroundR = fitted.y[-1]

            # calculate FWHM half heights on the left and right.

            halfMaxL = (peak_height - backgroundL) / 2 + backgroundL
            halfMaxR = (peak_height - backgroundR) / 2 + backgroundR

            def simple_interpolate(targetY: float, signal: XY) -> float:
                """Interpolates an XY profile to extract an x-value corresponding
                to a target y-value.

                Args:
                    targetY (float): Target y to interpolate x-value from.
                    signal (XY): XY signal to interpolate x-value from y-value.

                Returns:
                    float: x-value corresponding to targetY.
                """
                crossing_index = np.where(signal.y > targetY)[0][0]
                x1, x2 = signal.x[crossing_index - 1], signal.x[crossing_index]
                y1, y2 = signal.y[crossing_index - 1], signal.y[crossing_index]
                targetX = x1 + (targetY - y1) * (x2 - x1) / (y2 - y1)
                return targetX

            # interpolate x values corresponding to left and right FWHM half-heights.

            xL = simple_interpolate(halfMaxL, fitted)
            xR = simple_interpolate(halfMaxR, fitted[:, ::-1])

            # calculate FWHM and save fitted profile as instance attr.

            self.FWHM = xR - xL
            self.fitted = fitted

        def _fit_piecewise_sigmoid(self) -> XY:
            """Fits a piecewise sigmoid model to the raw signal for
            smoothing purposes.

            Returns:
                XY: Fitted sigmoid model for raw XY signal.
            """

            # get copy of raw signal for smoothing.
            smoothed = self.signal.copy()

            # apply basic smoothing using median and gaussian 1d convolution.
            k = round(len(smoothed.y) / 20)
            if k % 2 == 0:
                k += 1
            smoothed.y = medfilt(smoothed.y, k)
            smoothed.y = gaussian_filter1d(smoothed.y, round(k / 2.5))

            # find global maximum of raw signal.
            peaks, props = find_peaks(
                smoothed.y, height=0, prominence=np.max(smoothed.y).item() / 4
            )
            if len(peaks) == 0:
                self.handle_no_peaks_error()

            heights = props["peak_heights"]
            peak = peaks[np.argmax(heights)]

            def get_specific_sigmoid(
                wholeData: XY, fitStart: int, fitEnd: int
            ) -> Callable:
                """Fits a specific sigmoid across the whole data x-range.
                Fitting data is specified by args.

                Args:
                    wholeData (XY): XY object representing whole signal.
                    fitStart (int): index for start of data to use for sigmoid fitting.
                    fitEnd (int): index for end of data to use for sigmoid fitting.

                Returns:
                    Callable: Functional form of sigmoid for specific fitting range.
                """

                # select data to fit sigmoid from.
                fitData = wholeData[:, fitStart:fitEnd]

                # calculate initial guesses for sigmoid fitting parameters.
                A = np.max(fitData.y) - np.min(fitData.y)
                b = np.min(fitData.y)

                # k estimated from gradient calculations.
                dy = np.diff(fitData.y)
                dx = np.diff(fitData.x)
                dx[dx == 0] = 1e-6
                absDeriv = np.abs(dy / dx)
                absGradMax = np.max(absDeriv)
                k = np.sign(fitData.y[-1] - fitData.y[0]) * absGradMax * 0.5

                x0 = fitData.x[np.argmax(absDeriv)]

                p0 = [A, k, x0, b]

                def sigmoid(
                    x: float | np.ndarray, A: float, k: float, x0: float, b: float
                ) -> float:
                    """Functional form of sigmoid.

                    Args:
                        x (np.ndarray): input single value or x-array to get sigmoid y-val(s).
                        A (float): Multiplicative y-scalar param.
                        k (float): Sigmoid steepness param.
                        x0 (float): Sigmoid turning point param.
                        b (float): y-offset param.

                    Returns:
                        float | np.ndarray: Returned y-value(s) from sigmoid func.
                    """
                    exponent = np.clip(-k * (x - x0), -500, 500)
                    exp_term = np.exp(exponent)
                    return A / (1 + exp_term) + b

                # get fitted sigmoid parameters

                popt, _ = curve_fit(sigmoid, fitData.x, fitData.y, p0=p0, maxfev=10000)

                # return specific fitted sigmoid function for input data.

                def specific_sigmoid(x):
                    return sigmoid(x, *popt)

                return specific_sigmoid

            # get sigmoid functions for left and right sides of global maximum by fitting left and right sides of signal.

            sigmoidL_func = get_specific_sigmoid(smoothed, 0, peak)
            sigmoidR_func = get_specific_sigmoid(smoothed, peak, len(smoothed.x))

            # get an XY form of left and right functional sigmoids across whole x-range.

            sigmoidL = XY(smoothed.x, sigmoidL_func(smoothed.x))
            sigmoidR = XY(smoothed.x, sigmoidR_func(smoothed.x))

            def blending_weight(
                x: np.ndarray, transition_x: int, transition_width: int
            ) -> np.ndarray:
                """Generic function for defining a blending function (sigmoid method)

                Args:
                    x (np.ndarray): x-array covering whole x-range of functions to blend.
                    transition_x (int): x-value around which blending transition should happen.
                    transition_width (int): Approximate width of blending transition.

                Returns:
                    np.ndarray: blending function for input x-range.
                """
                return 1 / (1 + np.exp(-(x - transition_x) / transition_width))

            # get blending function for total x-range

            W = blending_weight(
                smoothed.x, peak, 1 / 20 * peak + 1 / 20 * (len(smoothed.x) - peak)
            )

            # blend XY representations of left and right sigmoid functions together using blending function.

            fitted = XY(smoothed.x, (1 - W) * sigmoidL.y + W * sigmoidR.y)

            return fitted

        def handle_no_peaks_error(self):
            """Handles error where no peaks are detected.
            Signal plot to be generated here (work for future)

            Raises:
                ValueError: No peaks detected error!
            """

            raise ValueError(
                "No peaks found in fitted signal due to poor phantom positioning."
            )

    def __init__(self, **kwargs):
        # Call constructor of HazenTask
        super().__init__(**kwargs)

        # Instantiate ACRObject class.
        self.ACR_obj = ACRObject(self.dcm_list)

    def run(self) -> dict:
        """Main function for performing slice width measurement
        using slice 1 from the ACR phantom image set.

        Returns:
            dict: results are returned in a standardised dictionary
                structure specifying the task name, input DICOM Series
                Description + SeriesNumber + InstanceNumber, task measurement
                key-value pairs, optionally path to the generated images
                for visualisation.
        """
        # Identify relevant slice, dcm and mask
        target_slice = 0
        dcm_slice_th = self.ACR_obj.dcms[target_slice]
        mask = self.ACR_obj.masks[target_slice]

        # Initialise results dictionary
        results = self.init_result_dict()
        results["file"] = self.img_desc(dcm_slice_th)

        try:
            # try to get slice thickness
            result = self.get_slice_thickness(dcm_slice_th, mask)
            results["measurement"] = {"slice width mm": round(result, 2)}
            print(f"{self.img_desc(dcm_slice_th)}: Slice thickness calculated.")

        except Exception as e:
            print(
                f"{self.img_desc(dcm_slice_th)}: Could not calculate slice thickness because of: {e}"
            )
            # traceback.print_exc(file=sys.stdout)

        # only return reports if requested
        if self.report:
            results["report_image"] = self.report_files

        return results

    def get_slice_thickness(self, dcm: FileDataset, mask: SliceMask) -> float:
        """Measure slice thickness.
        Identify the ramps, measure the line profile, measure the FWHM,
        and use this to calculate the slice thickness.

        Args:
            dcm (pydicom.Dataset): DICOM image object for slice thickness.
            mask (SliceMask): SliceMask object for slice thickness.

        Returns:
            float: measured slice thickness.
        """
        image = dcm.pixel_array

        # define interpolation factor for upscaling image
        interp_factor = 4

        # define pixel spacing in mm using interp_factor
        interp_pixel_mm = [dist / interp_factor for dist in self.ACR_obj.pixel_spacing]

        # resize image and mask for increased accuracy in profile line placement.
        new_dims = tuple([interp_factor * dim for dim in image.shape])
        image = cv2.resize(image, new_dims, interpolation=cv2.INTER_CUBIC)
        mask = mask.get_scaled_mask(interp_factor)

        # place profile lines on the image within the slice thickness insert.
        lines = self.place_lines(image, mask)

        # for each profile line, get the FWHM and multiply by combined mm/interpolation factor.
        for line in lines:
            line.get_FWHM()
            line.FWHM *= np.mean(interp_pixel_mm)

        # calculate slice thickness as per NEMA standard.
        slice_thickness = (
            0.2 * (lines[0].FWHM * lines[1].FWHM) / (lines[0].FWHM + lines[1].FWHM)
        )

        if self.report:

            fig, axes = plt.subplots(1, 3, figsize=(16, 8))
            axes[0].imshow(image)
            for i, line in enumerate(lines):
                axes[0].plot([line.start.x, line.end.x], [line.start.y, line.end.y])
                axes[i + 1].plot(
                    line.signal.x,
                    line.signal.y,
                    label="Raw signal",
                    alpha=0.25,
                    color=f"C{i}",
                )
                axes[i + 1].plot(
                    line.fitted.x,
                    line.fitted.y,
                    label="Fitted piecewise sigmoid",
                    color=f"C{i}",
                )
                axes[i + 1].legend(loc="lower right", bbox_to_anchor=(1, -0.2))

            axes[0].axis("off")

            axes[0].set_title("Plot showing placement of profile lines.")
            axes[1].set_title("Pixel profile across blue line.")
            axes[1].set_xlabel("Distance along blue line (pixels)")
            axes[1].set_ylabel("Pixel value")

            axes[2].set_title("Pixel profile across orange line.")
            axes[2].set_xlabel("Distance along orange line (pixels)")
            axes[2].set_ylabel("Pixel value")
            plt.tight_layout()

            image_path = os.path.realpath(
                os.path.join(
                    self.report_path, f"{self.img_desc(dcm)}_slice_thickness.png"
                )
            )

            fig.savefig(image_path, dpi=300)
            plt.close()
            self.report_files.append(image_path)

        return slice_thickness

    def place_lines(self, image: np.ndarray, mask: np.ndarray) -> list[Line]:
        """Places line on image within slice thickness insert.
        Works for a rotated phantom.

        Args:
            image (np.ndarray): Pixel array from DICOM image for slice thickness.
            mask (SliceMask): mask for slice thickness task.

        Returns:
            finalLines (list): A list of the two lines as Line objects.
        """

        insert = [c for c in mask.contours if is_slice_thickness_insert(c, mask.shape)][
            0
        ]

        # Create list of Point objects for the four corners of the contour
        corners = cv2.boxPoints(cv2.minAreaRect(insert))
        corners = [Point(*p) for p in corners]

        # Define short sides of contours by list of line objects
        corners = sorted(corners, key=lambda point: corners[0].get_distance_to(point))
        short_sides = [Line(*corners[:2]), Line(*corners[2:])]

        # Get sublines of short sides and force start point to be higher in y
        sublines = [line.get_subline(perc=30) for line in short_sides]
        for line in sublines:
            if line.start.y < line.end.y:
                line.point_swap()

        # Define connecting lines
        connecting_lines = [
            self.SignalLine(sublines[0].start, sublines[1].start),
            self.SignalLine(sublines[0].end, sublines[1].end),
        ]

        # Final lines are sublines of connecting lines
        final_lines = [line.get_subline(perc=95) for line in connecting_lines]
        for line in final_lines:
            line.get_signal(image)

        return final_lines
