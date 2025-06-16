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
import sys

import traceback
import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy.signal import find_peaks, medfilt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d


from hazenlib.HazenTask import HazenTask
from hazenlib.ACRObject import ACRObject
from hazenlib.utils import get_image_orientation, get_dicom_files
from hazenlib.contour_validation import is_slice_thickness_insert
from hazenlib.mask import Mask
from hazenlib.utils import Point, Line, XY


class ACRSliceThickness(HazenTask):
    """Slice width measurement class for DICOM images of the ACR phantom."""

    class SignalLine(Line):
        """Subclass of Line to implement functionality related to the ACR phantom's ramps"""

        def get_FWHM(self):
            """Calculates the FWHM by fitting a piecewise sigmoid to signal across line"""
            if not hasattr(self, "signal"):
                raise ValueError("Signal across line has not been computed!")

            fitted = self._fit_piecewise_sigmoid()

            _, props = find_peaks(
                fitted.y, height=0, prominence=np.ptp(fitted.y).item() / 4
            )
            peak_height = np.max(props["peak_heights"])

            backgroundL = fitted.y[0]
            backgroundR = fitted.y[-1]

            halfMaxL = (peak_height - backgroundL) / 2 + backgroundL
            halfMaxR = (peak_height - backgroundR) / 2 + backgroundR

            def simple_interpolate(targetY: float, signal: XY) -> float:
                crossing_index = np.where(signal.y > targetY)[0][0]
                x1, x2 = signal.x[crossing_index - 1], signal.x[crossing_index]
                y1, y2 = signal.y[crossing_index - 1], signal.y[crossing_index]
                targetX = x1 + (targetY - y1) * (x2 - x1) / (y2 - y1)
                return targetX

            xL = simple_interpolate(halfMaxL, fitted)
            xR = simple_interpolate(halfMaxR, fitted[:, ::-1])

            self.FWHM = xR - xL
            self.fitted = fitted

        def _fit_piecewise_sigmoid(self) -> XY:

            smoothed = self.signal.copy()
            k = round(len(smoothed.y) / 20)
            if k % 2 == 0:
                k += 1
            smoothed.y = medfilt(smoothed.y, k)
            smoothed.y = gaussian_filter1d(smoothed.y, round(k / 2.5))

            peaks, props = find_peaks(
                smoothed.y, height=0, prominence=np.max(smoothed.y).item() / 4
            )
            if len(peaks) == 0:
                self.handle_no_peaks_error()

            heights = props["peak_heights"]
            peak = peaks[np.argmax(heights)]

            def get_specific_sigmoid(wholeData: XY, fitStart: int, fitEnd: int) -> XY:
                fitData = wholeData[:, fitStart:fitEnd]
                A = np.max(fitData.y) - np.min(fitData.y)
                b = np.min(fitData.y)

                dy = np.diff(fitData.y)
                dx = np.diff(fitData.x)
                dx[dx == 0] = 1e-6
                absDeriv = np.abs(dy / dx)
                absGradMax = np.max(absDeriv)
                k = np.sign(fitData.y[-1] - fitData.y[0]) * absGradMax * 0.5
                x0 = fitData.x[np.argmax(absDeriv)]

                p0 = [A, k, x0, b]

                def sigmoid(x, A, k, x0, b):
                    exponent = np.clip(-k * (x - x0), -500, 500)
                    exp_term = np.exp(exponent)
                    return A / (1 + exp_term) + b

                popt, _ = curve_fit(sigmoid, fitData.x, fitData.y, p0=p0, maxfev=10000)

                def specific_sigmoid(x):
                    return sigmoid(x, *popt)

                return specific_sigmoid

            sigmoidL_func = get_specific_sigmoid(smoothed, 0, peak)
            sigmoidR_func = get_specific_sigmoid(smoothed, peak, len(smoothed.x))

            sigmoidL = XY(smoothed.x, sigmoidL_func(smoothed.x))
            sigmoidR = XY(smoothed.x, sigmoidR_func(smoothed.x))

            def blending_weight(x, transition_x, transition_width):
                return 1 / (1 + np.exp(-(x - transition_x) / transition_width))

            W = blending_weight(
                smoothed.x, peak, 1 / 20 * peak + 1 / 20 * (len(smoothed.x) - peak)
            )
            fitted = XY(smoothed.x, (1 - W) * sigmoidL.y + W * sigmoidR.y)

            return fitted

        def handle_no_peaks_error(self):
            # PLOT ERROR PLOT HERE
            raise ValueError(
                "No peaks found in fitted signal due to poor phantom positioning."
            )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialise ACR object
        self.ACR_obj = ACRObject(self.dcm_list)

    def run(self) -> dict:
        """Main function for performing slice width measurement
        using slice 1 from the ACR phantom image set.

        Returns:
            dict: results are returned in a standardised dictionary structure specifying the task name, input DICOM Series Description + SeriesNumber + InstanceNumber, task measurement key-value pairs, optionally path to the generated images for visualisation.
        """
        # Identify relevant slice
        slice_thickness_dcm = self.ACR_obj.dcms[0]

        # Initialise results dictionary
        results = self.init_result_dict()
        results["file"] = self.img_desc(slice_thickness_dcm)

        try:
            result = self.get_slice_thickness(slice_thickness_dcm)
            results["measurement"] = {"slice width mm": round(result, 2)}
            print(f"{self.img_desc(slice_thickness_dcm)}: Slice thickness calculated.")

        except Exception as e:
            print(
                f"{self.img_desc(slice_thickness_dcm)}: Could not calculate slice thickness because of: {e}"
            )
            # traceback.print_exc(file=sys.stdout)

        # only return reports if requested
        if self.report:
            results["report_image"] = self.report_files

        return results

    def get_slice_thickness(self, dcm):
        """Measure slice thickness. \n
        Identify the ramps, measure the line profile, measure the FWHM, and use this to calculate the slice thickness.

        Args:
            dcm (pydicom.Dataset): DICOM image object.

        Returns:
            float: measured slice thickness.
        """
        image = dcm.pixel_array
        mask = self.ACR_obj.masks[0]

        interp_factor = 4
        interp_pixel_mm = [dist / interp_factor for dist in self.ACR_obj.pixel_spacing]

        new_dims = tuple([interp_factor * dim for dim in image.shape])

        image = cv2.resize(image, new_dims, interpolation=cv2.INTER_CUBIC)
        mask = mask.get_scaled_mask(interp_factor)

        lines = self.place_lines(image, mask)

        for line in lines:
            line.get_FWHM()
            line.FWHM *= np.mean(interp_pixel_mm)
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

            fig.savefig(image_path, dpi=600)
            plt.close()
            self.report_files.append(image_path)

        return slice_thickness

    def place_lines(self, image: np.ndarray, mask: np.ndarray) -> list["Line"]:
        """Places line on image within ramps insert.
        Works for a rotated phantom.

        Args:
            image (np.ndarray): Pixel array from DICOM image.

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
