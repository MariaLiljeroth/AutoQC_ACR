"""
ACR SNR

Calculates the SNR for slice 7 (the uniformity slice) of the ACR phantom.

This script utilises the smoothed subtraction method described in McCann 2013:
A quick and robust method for measurement of signal-to-noise ratio in MRI, Phys. Med. Biol. 58 (2013) 3775:3790

and a standard subtraction SNR.

Created by Neil Heraghty (Adapted by Yassine Azma)

09/01/2023
"""

import os
import sys
import traceback
import pydicom

import numpy as np
from scipy import ndimage

import hazenlib.utils
from hazenlib.HazenTask import HazenTask
from hazenlib.ACRObject import ACRObject
from pydicom.pixel_data_handlers.util import apply_modality_lut


class ACRSNR(HazenTask):
    """Signal-to-noise ratio measurement class for DICOM images of the ACR phantom

    Inherits from HazenTask class
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ACR_obj = ACRObject(self.dcm_list)
        # measured slice width is expected to be a floating point number
        try:
            self.measured_slice_width = float(kwargs["measured_slice_width"])
        except:
            self.measured_slice_width = None

        # subtract is expected to be a path to a folder
        try:
            if os.path.isdir(kwargs["subtract"]):
                self.subtract = kwargs["subtract"]
        except:
            self.subtract = None

    def run(self) -> dict:
        """Main function for performing SNR measurement
        using slice 7 from the ACR phantom image set

        Notes:
            using the smoothing method by default or the subtraction method if a second set of images are provided (in a separate folder)

        Returns:
            dict: results are returned in a standardised dictionary structure specifying the task name, input DICOM Series Description + SeriesNumber + InstanceNumber, task measurement key-value pairs, optionally path to the generated images for visualisation
        """
        # Identify relevant slice
        dcm_snr = self.ACR_obj.dcms[4]
        mask_snr = self.ACR_obj.masks[4]

        # Initialise results dictionary
        results = self.init_result_dict()

        # SINGLE METHOD (SMOOTHING)
        if self.subtract is None:
            try:
                results["file"] = self.img_desc(dcm_snr)
                snr, normalised_snr, signal, noise, col, row = self.snr_by_smoothing(
                    dcm_snr, mask_snr, self.measured_slice_width
                )
                results["measurement"]["snr by smoothing"] = {
                    "measured": round(snr, 2),
                    "normalised": round(normalised_snr, 2),
                    "signal": signal,
                    "noise": noise,
                    "centre y": row,
                    "centre x": col,
                }
                print(f"{self.img_desc(dcm_snr)}: SNR calculated.")

            except Exception as e:
                print(
                    f"{self.img_desc(dcm_snr)}: Could not calculate SNR because of: {e}"
                )
                # traceback.print_exc(file=sys.stdout)

        # SUBTRACTION METHOD
        else:
            # Get the absolute path to all FILES found in the directory provided
            filepaths = [
                os.path.join(self.subtract, f)
                for f in os.listdir(self.subtract)
                if os.path.isfile(os.path.join(self.subtract, f))
            ]
            data2 = [pydicom.dcmread(dicom) for dicom in filepaths]

            ACR_obj_2 = ACRObject(data2)
            dcm_snr2 = ACR_obj_2.dcms[4]
            mask_snr2 = ACR_obj_2.masks[4]
            results["file"] = [self.img_desc(dcm_snr), self.img_desc(dcm_snr2)]
            try:
                snr, normalised_snr = self.snr_by_subtraction(
                    dcm_snr, mask_snr, dcm_snr2, mask_snr2, self.measured_slice_width
                )

                results["measurement"]["snr by subtraction"] = {
                    "measured": round(snr, 2),
                    "normalised": round(normalised_snr, 2),
                }
                print(f"{self.img_desc(dcm_snr)}: SNR calculated.")
            except Exception as e:
                print(
                    f"Could not calculate the SNR for {self.img_desc(dcm_snr)} and "
                    f"{self.img_desc(dcm_snr2)} because of : {e}"
                )
                # traceback.print_exc(file=sys.stdout)

        # only return reports if requested
        if self.report:
            results["report_image"] = self.report_files

        return results

    def snr_by_smoothing(
        self, dcm: pydicom.Dataset, mask: "SliceMask", measured_slice_width=None
    ) -> float:
        """Calculate signal to noise ratio based on smoothing method

        Args:
            dcm (pydicom.Dataset): DICOM image object
            measured_slice_width (float, optional): Provide the true slice width for the set of images. Defaults to None.

        Returns:
            float: normalised_snr
        """
        centre = mask.centre
        radius = mask.radius

        signal = [
            np.mean(roi)
            for roi in self.get_roi_samples(
                ax=None, dcm=dcm, centre=centre, phantom_radius=radius
            )
        ]
        noise = [
            np.std(roi, ddof=1)
            for roi in self.get_roi_samples(
                ax=None,
                dcm=dcm,
                centre=centre,
                phantom_radius=radius,
                place_in_background=True,
            )
        ]
        # note no root_2 factor in noise for smoothed subtraction (one image) method, replicating Matlab approach and
        # McCann 2013

        # 0.655 factor included to correct for un-gaussian noise distribution
        snr = np.mean(signal) / (np.mean(noise) / 0.655)

        normalised_snr = snr * self.get_normalised_snr_factor(dcm, measured_slice_width)

        if self.report:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1)
            fig.set_size_inches(8, 16)
            fig.tight_layout(pad=4)

            axes[0].imshow(dcm.pixel_array)
            axes[0].scatter(*centre, c="red")
            axes[0].set_title("Centroid Location")
            circle1 = plt.Circle(centre, mask.radius, color="r", fill=False)
            axes[0].add_patch(circle1)

            axes[1].set_title("Smoothed Noise Image")
            axes[1].imshow(dcm.pixel_array, cmap="gray")
            self.get_roi_samples(axes[1], dcm, centre, radius)
            self.get_roi_samples(axes[1], dcm, centre, radius, place_in_background=True)

            img_path = os.path.realpath(
                os.path.join(self.report_path, f"{self.img_desc(dcm)}_smoothing.png")
            )
            fig.savefig(img_path)
            plt.close()
            self.report_files.append(img_path)

        return (
            snr,
            normalised_snr,
            [round(elem, 1) for elem in signal],
            [round(elem, 2) for elem in noise],
            *centre,
        )

    def snr_by_subtraction(
        self,
        dcm1: pydicom.Dataset,
        mask_snr: "SliceMask",
        dcm2: pydicom.Dataset,
        mask_snr2: "SliceMask",
        measured_slice_width=None,
    ) -> float:
        """Calculate signal to noise ratio based on subtraction method

        Args:
            dcm1 (pydicom.Dataset): DICOM image object to calculate signal
            dcm2 (pydicom.Dataset): DICOM image object to calculate noise
            measured_slice_width (float, optional): Provide the true slice width for the set of images. Defaults to None.

        Returns:
            float: normalised_snr
        """
        centre1 = mask_snr.centre
        centre2 = mask_snr2.centre

        radius1 = mask_snr.radius
        radius2 = mask_snr.radius

        difference = np.subtract(
            apply_modality_lut(dcm1.pixel_array, dcm1).astype("int"),
            apply_modality_lut(dcm2.pixel_array, dcm2).astype(
                "int"
            ),  # dcm1.pixel_array.astype("int"), dcm2.pixel_array.astype("int")
        )

        signal = [
            np.mean(roi)
            for roi in self.get_roi_samples(
                ax=None, dcm=dcm1, centre=centre1, phantom_radius=radius1
            )
        ]
        noise = np.divide(
            [
                np.std(roi, ddof=1)
                for roi in self.get_roi_samples(
                    ax=None, dcm=difference, centre=centre2, phantom_radius=radius2
                )
            ],
            np.sqrt(2),
        )
        snr = np.mean(signal) / np.mean(noise)

        normalised_snr = snr * self.get_normalised_snr_factor(
            dcm1, measured_slice_width
        )

        if self.report:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1)
            fig.set_size_inches(8, 16)
            fig.tight_layout(pad=4)

            axes[0].imshow(dcm1.pixel_array)
            axes[0].scatter(*centre1, c="red")
            axes[0].axis("off")
            axes[0].set_title("Centroid Location")

            axes[1].set_title("Difference Image")
            axes[1].imshow(
                difference,
                cmap="gray",
            )
            self.get_roi_samples(axes[1], dcm1, centre1, radius1)
            axes[1].axis("off")

            img_path = os.path.realpath(
                os.path.join(
                    self.report_path, f"{self.img_desc(dcm1)}_snr_subtraction.png"
                )
            )
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return snr, normalised_snr

    def get_roi_samples(
        self,
        ax,
        dcm: pydicom.Dataset or np.ndarray,
        centre: tuple,
        phantom_radius: int | float,
        place_in_background: bool = False,
    ) -> list:
        """Identify regions of interest

        Args:
            ax (matplotlib.pyplot.subplots): matplotlib axis for visualisation
            dcm (pydicom.Dataset or np.ndarray): DICOM image object, or its pixel array
            centre_col (int): x coordinate of the centre
            centre_row (int): y coordinate of the centre

        Returns:
            list of np.array: subsets of the original pixel array
        """
        if type(dcm) == np.ndarray:
            data = dcm
        else:
            data = apply_modality_lut(dcm.pixel_array, dcm).astype(
                "int"
            )  # dcm.pixel_array

        centre_col, centre_row = centre

        roi_size = phantom_radius // 4
        roi_size += roi_size % 2
        x_shift, y_shift = [phantom_radius // 2.5] * 2

        if place_in_background:
            rows, cols = data.shape
            pad = np.mean([rows, cols]) // 20
            centres = (
                (pad + roi_size / 2, pad + roi_size / 2),
                (cols - pad - roi_size / 2, pad + roi_size / 2),
                (cols - pad - roi_size / 2, rows - pad - roi_size / 2),
                (pad + roi_size / 2, rows - pad - roi_size / 2),
            )

        else:
            centres = (
                [centre_col, centre_row],
                [centre_col - x_shift, centre_row - y_shift],
                [centre_col - x_shift, centre_row + y_shift],
                [centre_col + x_shift, centre_row - y_shift],
                [centre_col + x_shift, centre_row + y_shift],
            )

        sample = [
            data[
                int(c[1] - roi_size / 2) : int(c[1] + roi_size / 2),
                int(c[0] - roi_size / 2) : int(c[0] + roi_size / 2),
            ]
            for c in centres
        ]

        if ax:
            from matplotlib.patches import Rectangle
            from matplotlib.collections import PatchCollection

            # for patches: [column/x, row/y] format

            rects = [
                Rectangle(
                    (c[0] - roi_size // 2, c[1] - roi_size // 2), roi_size, roi_size
                )
                for c in centres
            ]
            pc = PatchCollection(
                rects, edgecolors="red", facecolors="None", label="ROIs"
            )
            ax.add_collection(pc)

        return sample

    def get_normalised_snr_factor(self, dcm, measured_slice_width=None) -> float:
        """Calculate the normalisation factor to be applied

        Args:
            dcm (pydicom.Dataset): DICOM image object
            measured_slice_width (float, optional): Provide the true slice width for the set of images. Defaults to None.

        Returns:
            float: normalisation factor
        """
        dx, dy = hazenlib.utils.get_pixel_size(dcm)
        bandwidth = hazenlib.utils.get_bandwidth(dcm)
        TR = hazenlib.utils.get_TR(dcm)
        rows = hazenlib.utils.get_rows(dcm)
        columns = hazenlib.utils.get_columns(dcm)

        if measured_slice_width:
            slice_thickness = measured_slice_width
        else:
            slice_thickness = hazenlib.utils.get_slice_thickness(dcm)

        averages = hazenlib.utils.get_average(dcm)
        bandwidth_factor = np.sqrt((bandwidth * columns / 2) / 1000) / np.sqrt(30)
        voxel_factor = 1 / (0.001 * dx * dy * slice_thickness)

        normalised_snr_factor = (
            bandwidth_factor
            * voxel_factor
            * (1 / (np.sqrt(averages * rows * (TR / 1000))))
        )
        return normalised_snr_factor

    # def filtered_image(self, dcm: pydicom.Dataset) -> np.array:
    #     """Apply filtering to a pixel array (image)

    #     Notes:
    #         Performs a 2D convolution (for filtering images)
    #         uses uniform_filter SciPy function

    #     Args:
    #         dcm (pydicom.Dataset): DICOM image object

    #     Returns:
    #         np.array: pixel array of the filtered image
    #     """
    #     # a = dcm.pixel_array.astype("int")
    #     a = apply_modality_lut(dcm.pixel_array, dcm).astype("int")
    #     # filter size = 9, following MATLAB code and McCann 2013 paper for head coil, although note McCann 2013
    #     # recommends 25x25 for body coil.
    #     filtered_array = ndimage.uniform_filter(a, 9, mode="constant")
    #     return filtered_array

    # def get_noise_image(self, dcm: pydicom.Dataset) -> np.array:
    #     """Get noise image by subtracting the filtered image from the original pixel array

    #     Notes:
    #         Separates the image noise by smoothing the image and subtracting the smoothed image from the original.

    #     Args:
    #         dcm (pydicom.Dataset): DICOM image object

    #     Returns:
    #         np.array: pixel array representing the image noise
    #     """
    #     # a = dcm.pixel_array.astype("int")
    #     a = apply_modality_lut(dcm.pixel_array, dcm).astype("int")

    #     # Convolve image with boxcar/uniform kernel
    #     imsmoothed = self.filtered_image(dcm)

    #     # Subtract smoothed array from original
    #     imnoise = a - imsmoothed

    #     return imnoise
