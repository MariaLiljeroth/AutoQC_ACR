"""
acr_snr.py

Calculates the SNR for the most uniform slice of the ACR phantom image set.

This script implements both traditional and subtraction methods for SNR calculation.

Created by Neil Heraghty (Adapted by Yassine Azma) (Then adapted by Nathan Crossley for RSCH local purposes, 2025)

09/01/2023

"""

from typing import Self
import os
import numpy as np

from pydicom import Dataset, dcmread
from pydicom.pixel_data_handlers.util import apply_modality_lut

from src.backend.hazen.hazenlib import utils
from src.backend.hazen.hazenlib.HazenTask import HazenTask
from src.backend.hazen.hazenlib.ACRObject import ACRObject
from src.backend.hazen.hazenlib.masking_tools.slice_mask import SliceMask


class ACRSNR(HazenTask):
    """Subclass of HazenTask that contains code relating to calculating
    the SNR of dcm images in the ACR phantom image set.
    """

    def __init__(self, **kwargs):

        # Call initialiser of HazenTask
        super().__init__(**kwargs)

        # Instantiate ACRObject class using dcm list passed in within kwargs
        self.ACR_obj = ACRObject(self.dcm_list)

        # measured slice width is expected to be a floating point number
        try:
            self.measured_slice_width = float(kwargs["measured_slice_width"])
        except:
            self.measured_slice_width = None

        # subtract kwarg is expected to be a path to a folder
        try:
            if os.path.isdir(kwargs["subtract"]):
                self.subtract = kwargs["subtract"]
        except:
            self.subtract = None

    def run(self) -> dict:
        """Entry function for performing SNR measurement
        using most uniform slice from the ACR phantom image set

        Notes:
            using the smoothing method by default or the subtraction method if a second set of images are provided (in a separate folder)

        Returns:
            dict: results are returned in a standardised dictionary
                structure specifying the task name, input DICOM Series
                Description + SeriesNumber + InstanceNumber, task measurement
                key-value pairs, optionally path to the generated images for
                visualisation.
        """
        # Identify relevant slice, dcm and mask
        target_slice = self.ACR_obj.most_uniform_slice
        dcm_snr = self.ACR_obj.dcms[target_slice]
        mask_snr = self.ACR_obj.masks[target_slice]

        # Initialise results dictionary
        results = self.init_result_dict()

        # SINGLE METHOD (SMOOTHING)
        if self.subtract is None:
            try:

                # append dcm description to results
                results["file"] = self.img_desc(dcm_snr)

                # try to calculate snr by smoothing (single dcm) method
                snr, normalised_snr, signal, noise, col, row = self.snr_by_smoothing(
                    dcm_snr, mask_snr, self.measured_slice_width
                )

                # append results to results dict
                results["measurement"]["snr by smoothing"] = {
                    "measured": round(snr, 2),
                    "normalised": round(normalised_snr, 2),
                    "signal": signal,
                    "noise": noise,
                    "centre y": row,
                    "centre x": col,
                }

                # signal to user that SNR has been calculated for a particular dcm
                print(f"{self.img_desc(dcm_snr)}: SNR calculated.")

            except Exception as e:

                # alert the user that the SNR could not be calculated and why
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

            # read the dcm data for all files (all assumed to be dcm)
            data2 = [dcmread(dicom) for dicom in filepaths]

            # Instantiate a second ACRObject object using second set of dcms
            ACR_obj_2 = ACRObject(data2)

            # Identify relevant slice for this second set of dcms
            target_slice_2 = ACR_obj_2.most_uniform_slice
            dcm_snr2 = ACR_obj_2.dcms[target_slice_2]
            mask_snr2 = ACR_obj_2.masks[target_slice_2]

            # append dcm descriptions to results dict
            results["file"] = [self.img_desc(dcm_snr), self.img_desc(dcm_snr2)]
            try:

                # try and get snr by subtraction method
                snr, normalised_snr = self.snr_by_subtraction(
                    dcm_snr, mask_snr, dcm_snr2, mask_snr2, self.measured_slice_width
                )

                # append results to results dict
                results["measurement"]["snr by subtraction"] = {
                    "measured": round(snr, 2),
                    "normalised": round(normalised_snr, 2),
                }

                # inform the user that the SNR was calculated for a particular dcm
                print(f"{self.img_desc(dcm_snr)}: SNR calculated.")

            except Exception as e:

                # alert the user that the SNR could not be calculated for a particular dcm and why
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
        self, dcm: Dataset, mask: SliceMask, measured_slice_width: float = None
    ) -> tuple[float, float, list[float], list[float], float, float]:
        """
        Calculate SNR based on single dcm method

        Args:
            dcm (Dataset): Selected dcm for the SNR by smoothing calculation
            mask (SliceMask): The binary mask associated with this dcm
            measured_slice_width (float, optional): Provide the true slice width for the set of images. Defaults to None.

        Returns:
            tuple[float, float, list[float], list[float], float, float]: Values for SNR, normalised SNR,
                signal list, noise list, centre-x and centre-y.
        """

        # get centre and radius of mask
        centre = mask.centre
        radius = mask.radius

        # get signal values from mean of ROIs placed within uniformity slice
        signal = [
            np.mean(roi)
            for roi in self.get_roi_samples(
                ax=None, dcm=dcm, centre=centre, phantom_radius=radius
            )
        ]

        # get noise values from mean of ROIs placed within background
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

        # get normalised snr by multipling by calculated factor
        normalised_snr = snr * self.get_normalised_snr_factor(dcm, measured_slice_width)

        if self.report:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1)
            fig.set_size_inches(8, 16)
            fig.tight_layout(pad=4)

            # plot dcm used for snr calculations and detected centre
            axes[0].imshow(dcm.pixel_array)
            axes[0].scatter(*centre, c="red")
            axes[0].set_title("Centroid Location")
            circle1 = plt.Circle(centre, mask.radius, color="r", fill=False)
            axes[0].add_patch(circle1)
            axes[0].axis("off")

            # show placement of signal and noise ROIs on original dcm pixel array
            axes[1].set_title("ROI Placement for Standard SNR")
            axes[1].imshow(dcm.pixel_array)
            self.get_roi_samples(axes[1], dcm, centre, radius)
            self.get_roi_samples(axes[1], dcm, centre, radius, place_in_background=True)
            axes[1].axis("off")

            # work out path for saving plots
            img_path = os.path.realpath(
                os.path.join(self.report_path, f"{self.img_desc(dcm)}_smoothing.png")
            )

            # save plot at path and store path as an attr
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
        dcm1: Dataset,
        mask_snr: SliceMask,
        dcm2: Dataset,
        mask_snr2: SliceMask,
        measured_slice_width=None,
    ) -> float:
        """Calculate SNR based on subtraction method

        Args:
            dcm1 (Dataset): Dcm to calculate signal.
            dcm2 (Dataset): Second dcm to use for subtraction to calculate noise.
            measured_slice_width (float, optional): Provide the true slice width for the set of images. Defaults to None.

        Returns:
            tuple[float]: SNR and normalised SNR
        """

        # calculate centre of each mask and the average centre
        centre1 = mask_snr.centre
        centre2 = mask_snr2.centre
        centre_av = [np.mean([a, b]) for a, b in zip(centre1, centre2)]

        # calculate radius of each mask and the average radius
        radius1 = mask_snr.radius
        radius2 = mask_snr.radius
        radius_av = np.mean([radius1, radius2])

        # get difference image to simulate noise
        difference = np.subtract(
            apply_modality_lut(dcm1.pixel_array, dcm1).astype("int"),
            apply_modality_lut(dcm2.pixel_array, dcm2).astype(
                "int"
            ),  # dcm1.pixel_array.astype("int"), dcm2.pixel_array.astype("int")
        )

        # get signal values by taking mean within ROIs place within the uniformity slice of dcm 1
        signal = [
            np.mean(roi)
            for roi in self.get_roi_samples(
                ax=None, dcm=dcm1, centre=centre1, phantom_radius=radius1
            )
        ]

        # get signal values by taking mean within ROIs place within the "noise" of the difference image
        noise = np.divide(
            [
                np.std(roi, ddof=1)
                for roi in self.get_roi_samples(
                    ax=None, dcm=difference, centre=centre_av, phantom_radius=radius_av
                )
            ],
            np.sqrt(2),
        )

        # calculate SNR
        snr = np.mean(signal) / np.mean(noise)

        # calculate normalised SNR by multiplication by calculated factor
        normalised_snr = snr * self.get_normalised_snr_factor(
            dcm1, measured_slice_width
        )

        if self.report:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1)
            fig.set_size_inches(8, 16)
            fig.tight_layout(pad=4)

            # show first dcm and detected centre
            axes[0].imshow(dcm1.pixel_array)
            axes[0].scatter(*centre1, c="red")
            axes[0].axis("off")
            axes[0].set_title("Centroid Location")
            circle1 = plt.Circle(centre1, radius1, color="r", fill=False)
            axes[0].add_patch(circle1)

            # show ROI placement for signal (within original image)
            axes[1].set_title("ROI Placement in Original Image")
            axes[1].imshow(dcm1.pixel_array)
            self.get_roi_samples(axes[1], dcm1, centre1, radius1)
            axes[1].axis("off")

            # show ROI placement for nosie (within difference image)
            axes[2].set_title("ROI Placement in Difference Image")
            axes[2].imshow(difference)
            self.get_roi_samples(axes[2], difference, centre_av, radius_av)
            axes[2].axis("off")

            # work out path for saving figure
            img_path = os.path.realpath(
                os.path.join(
                    self.report_path, f"{self.img_desc(dcm1)}_snr_subtraction.png"
                )
            )

            # save the figure and store path as an instance attr
            fig.savefig(img_path, dpi=300)
            self.report_files.append(img_path)

        return snr, normalised_snr

    def get_roi_samples(
        self: Self,
        ax,
        dcm: Dataset | np.ndarray,
        centre: tuple,
        phantom_radius: int | float,
        place_in_background: bool = False,
    ) -> list:
        """Position ROIs within unformity slice or background and
        extract pixel values.

        Args:
            ax (matplotlib.pyplot.subplots): matplotlib axis for visualisation.
            dcm (Dataset | np.ndarray): DICOM image object, or its pixel array
            centre (tuple): phantom/mask centre
            phantom_radius (int | float): calculated phantom radius
            place_in_background (bool): True if ROIs should be placed in background, False otherwise

        Returns:
            list[np.ndarray]: subsets of the original pixel array (extract within ROIs)
        """

        # if dcm, get pixel array, otherwise pass
        if type(dcm) == np.ndarray:
            data = dcm
        else:
            data = apply_modality_lut(dcm.pixel_array, dcm).astype(
                "int"
            )  # dcm.pixel_array

        # store references to col and row corresponding to centre
        centre_col, centre_row = centre

        # determine the roi_size, roi x-shift and roi y-shift, in pixels
        roi_size = phantom_radius // 4
        roi_size += roi_size % 2
        x_shift, y_shift = [phantom_radius // 2.5] * 2

        if place_in_background:

            # position ROI centres in background (4 ROIs)
            rows, cols = data.shape
            pad = np.mean([rows, cols]) // 20
            centres = (
                (pad + roi_size / 2, pad + roi_size / 2),
                (cols - pad - roi_size / 2, pad + roi_size / 2),
                (cols - pad - roi_size / 2, rows - pad - roi_size / 2),
                (pad + roi_size / 2, rows - pad - roi_size / 2),
            )

        else:

            # position ROI centres within phantom (5 ROIs)
            centres = (
                [centre_col, centre_row],
                [centre_col - x_shift, centre_row - y_shift],
                [centre_col - x_shift, centre_row + y_shift],
                [centre_col + x_shift, centre_row - y_shift],
                [centre_col + x_shift, centre_row + y_shift],
            )

        # extract np.ndarray objects for each ROI
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

            # if matplotlib axis is passed, construct Rectangle patches to visually represent ROIs
            rects = [
                Rectangle(
                    (c[0] - roi_size // 2, c[1] - roi_size // 2), roi_size, roi_size
                )
                for c in centres
            ]

            # add patches to axis
            pc = PatchCollection(
                rects, edgecolors="red", facecolors="None", label="ROIs"
            )
            ax.add_collection(pc)

        return sample

    def get_normalised_snr_factor(
        self, dcm: Dataset, measured_slice_width=None
    ) -> float:
        """Calculate the normalisation factor to be applied

        Args:
            dcm (Dataset): DICOM image object
            measured_slice_width (float, optional): Provide the true slice width for the set of images. Defaults to None.

        Returns:
            float: normalisation factor
        """
        dx, dy = utils.get_pixel_size(dcm)
        bandwidth = utils.get_bandwidth(dcm)
        TR = utils.get_TR(dcm)
        rows = utils.get_rows(dcm)
        columns = utils.get_columns(dcm)

        if measured_slice_width:
            slice_thickness = measured_slice_width
        else:
            slice_thickness = utils.get_slice_thickness(dcm)

        averages = utils.get_average(dcm)
        bandwidth_factor = np.sqrt((bandwidth * columns / 2) / 1000) / np.sqrt(30)
        voxel_factor = 1 / (0.001 * dx * dy * slice_thickness)

        normalised_snr_factor = (
            bandwidth_factor
            * voxel_factor
            * (1 / (np.sqrt(averages * rows * (TR / 1000))))
        )
        return normalised_snr_factor

    # def filtered_image(self, dcm: Dataset) -> np.array:
    #     """Apply filtering to a pixel array (image)

    #     Notes:
    #         Performs a 2D convolution (for filtering images)
    #         uses uniform_filter SciPy function

    #     Args:
    #         dcm (Dataset): DICOM image object

    #     Returns:
    #         np.array: pixel array of the filtered image
    #     """
    #     # a = dcm.pixel_array.astype("int")
    #     a = apply_modality_lut(dcm.pixel_array, dcm).astype("int")
    #     # filter size = 9, following MATLAB code and McCann 2013 paper for head coil, although note McCann 2013
    #     # recommends 25x25 for body coil.
    #     filtered_array = ndimage.uniform_filter(a, 9, mode="constant")
    #     return filtered_array

    # def get_noise_image(self, dcm: Dataset) -> np.array:
    #     """Get noise image by subtracting the filtered image from the original pixel array

    #     Notes:
    #         Separates the image noise by smoothing the image and subtracting the smoothed image from the original.

    #     Args:
    #         dcm (Dataset): DICOM image object

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
