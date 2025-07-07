"""
ACR Uniformity

https://www.acraccreditation.org/-/media/acraccreditation/documents/mri/largephantomguidance.pdf

Calculates the percentage integral uniformity for slice 7 of the ACR phantom.

This script calculates the percentage integral uniformity in accordance with the ACR Guidance.
This is done by first defining a large 200cm2 ROI before placing 1cm2 ROIs at every pixel within
the large ROI. At each point, the mean of the 1cm2 ROI is calculated. The ROIs with the maximum and
minimum mean value are used to calculate the integral uniformity. The results are also visualised.

Created by Yassine Azma (Adapted by Nathan Crossley for local RSCH purposes, 2025)
yassine.azma@rmh.nhs.uk

13/01/2022
"""

import os
import numpy as np
from pydicom import Dataset

from backend.hazen.hazenlib.HazenTask import HazenTask
from backend.hazen.hazenlib.ACRObject import ACRObject
from backend.hazen.hazenlib.masking_tools.slice_mask import SliceMask


class ACRUniformity(HazenTask):
    """Subclass of Hazentask that contains code relating to calculating
    the percentage integral uniformity of dcm images of the ACR phantom image set.
    """

    def __init__(self, **kwargs):
        # Call initialiser of HazenTask
        super().__init__(**kwargs)

        # Instantiate ACRObject class using dcm list passed in within kwargs
        self.ACR_obj = ACRObject(self.dcm_list, kwargs)

    def run(self) -> dict:
        """Entrypoint function to trigger the uniformity calculation, using the
        forst uniform slice of the ACR phantom image set.

        Returns:
            dict: results are returned in a standardised dictionary
                structure specifying the task name, input DICOM Series
                Description + SeriesNumber + InstanceNumber, task measurement
                key-value pairs, optionally path to the generated images
                for visualisation
        """
        # Identify relevant slice, dcm and mask
        target_slice = self.ACR_obj.most_uniform_slice
        dcm_unif = self.ACR_obj.dcms[target_slice]
        mask_unif = self.ACR_obj.masks[target_slice]

        # Initialise results dictionary and add image description
        results = self.init_result_dict()
        results["file"] = self.img_desc(dcm_unif)

        try:

            # get uniformity measurement of chosen dcm and mask.
            # max and min values and positions also reported.
            unif, max_roi, min_roi, max_pos, min_pos = self.get_integral_uniformity(
                dcm_unif, mask_unif
            )

            # append results to results dict
            results["measurement"] = {
                "integral uniformity %": round(unif, 2),
                "max roi": round(max_roi, 1),
                "min roi": round(min_roi, 1),
                "max pos": max_pos,
                "min pos": min_pos,
            }

            # signal to user that uniformity has been calculated for given dcm
            print(
                f"{self.img_desc(dcm_unif)}: Percentage integral uniformity calculated."
            )

        except Exception as e:

            # alert the user that uniformity could not be calculated and why
            print(
                f"{self.img_desc(dcm_unif)}: Could not calculate percentage integral uniformity because of: {e}"
            )
            # traceback.print_exc(file=sys.stdout)

        # only return reports if requested
        if self.report:
            results["report_image"] = self.report_files

        return results

    def get_integral_uniformity(
        self, dcm: Dataset, mask_unif: SliceMask
    ) -> tuple[float, float, float, tuple[int], tuple[int]]:
        """Measure percentage integral uniformity of the provided dcm.
        Define a test circular mask around slice mask's centre. Calculate
        the min and max pixel values and then calculate percentage integral
        uniformity. Fallback sliding window method is provided.

        Args:
            dcm (Dataset): Dcm of chosen ACR slice for uniformity task.
            mask_unif (SliceMask): Corresponding mask of chosen dcm.

        Returns:
            tuple[float, float, float, tuple[int], tuple[int]]: Returns percentage
                integral uniformity, max and min vals as well as positions of max and min.
        """

        img = dcm.pixel_array
        from pydicom.pixel_data_handlers.util import apply_modality_lut

        img = apply_modality_lut(dcm.pixel_array, dcm).astype("uint16")

        # try to get in-plane resolution
        if "PixelSpacing" in dcm:
            res = dcm.PixelSpacing
        else:
            import hazenlib.utils

            res = hazenlib.utils.GetDicomTag(dcm, (0x28, 0x30))

        # Define large radius that produces ~200cm2 ROI
        r_large = np.ceil(80 / res[0]).astype(int)

        # Define small radius that produces ~200cm2 ROI
        r_small = np.ceil(np.sqrt(100 / np.pi) / res[0]).astype(int)

        if self.ACR_obj.MediumACRPhantom == True:
            # Making it a 90% smaller than 160cm^2 (16000mm^2) to avoid the bit at the top
            r_large = np.ceil(np.sqrt(16000 * 0.90 / np.pi) / res[0]).astype(int)

        # Offset distance for rectangular void at top of phantom
        d_void = np.ceil(5 / res[0]).astype(int)

        # Store image dims
        dims = img.shape

        # get chosen masks's cenre
        cxy = mask_unif.centre

        # construct dummy mask at centroid and save coords of mask
        base_mask = ACRObject.circular_mask((cxy[0], cxy[1] + d_void), r_small, dims)
        coords = np.nonzero(base_mask)

        # get large ROI by circular mask
        lroi = self.ACR_obj.circular_mask([cxy[0], cxy[1] + d_void], r_large, dims)

        # get masked image using large circular test mask
        img_masked = lroi * img

        # get half max pixel value for non-zero masked image
        half_max = np.percentile(img_masked[np.nonzero(img_masked)], 50)

        # split masked image into upper and lower halves, according to pixel value
        min_image = img_masked * (img_masked < half_max)
        max_image = img_masked * (img_masked > half_max)

        # store coords of upper and lower halves of masked image
        min_rows, min_cols = np.nonzero(min_image)[0], np.nonzero(min_image)[1]
        max_rows, max_cols = np.nonzero(max_image)[0], np.nonzero(max_image)[1]

        # construct array to receive mean vals
        mean_array = np.zeros(img_masked.shape)

        def uniformity_iterator(masked_image, sample_mask, rows, cols):
            """Iterate through a pixel array and determine mean value

            Args:
                masked_image (np.array): subset of pixel array
                sample_mask (np.array): _description_
                rows (np.array): 1D array
                cols (np.array): 1D array

            Returns:
                np.array: array of mean values
            """
            coords = np.nonzero(sample_mask)  # Coordinates of mask
            for idx, (row, col) in enumerate(zip(rows, cols)):
                centre = [row, col]
                translate_mask = [
                    np.intp(coords[0] + centre[0] - cxy[0] - d_void),
                    np.intp(coords[1] + centre[1] - cxy[1]),
                ]
                values = masked_image[translate_mask[0], translate_mask[1]]
                if np.count_nonzero(values) < np.count_nonzero(sample_mask):
                    mean_val = 0
                else:
                    mean_val = np.mean(values[np.nonzero(values)])

                mean_array[row, col] = mean_val

            return mean_array

        min_data = uniformity_iterator(min_image, base_mask, min_rows, min_cols)
        max_data = uniformity_iterator(max_image, base_mask, max_rows, max_cols)

        if (
            np.all(min_data == 0) == 0 and np.all(max_data == 0) == 0
        ):  # if the array is all 0s skip
            sig_max = np.max(max_data)
            sig_min = np.min(min_data[np.nonzero(min_data)])
            max_loc = np.where(max_data == sig_max)
            min_loc = np.where(min_data == sig_min)

            max_loc = (max_loc[0][0], max_loc[1][0])
            min_loc = (min_loc[0][0], min_loc[1][0])

        else:  # If the image doenst give a nice blob (like the ACR wants, revert to jsut a asliding window approach which should work regardless of the noise) but its not really the ACR way...
            print(
                "Reverting to sliding window over whole image, this sometimes happens when there are quite noisy images!"
            )
            rows, cols = np.nonzero(img_masked)[0], np.nonzero(img_masked)[1]
            mean_array = np.zeros(img_masked.shape)

            coords = np.nonzero(base_mask)  # Coordinates of mask
            for idx, (row, col) in enumerate(zip(rows, cols)):
                centre = [row, col]
                translate_mask = [
                    np.intp(coords[0] + centre[0] - cxy[0] - d_void),
                    np.intp(coords[1] + centre[1] - cxy[1]),
                ]
                values = img_masked[tuple(translate_mask)]
                if (
                    np.count_nonzero(values == 0) == 0
                ):  # Incase we clip out of hte mask so lets just dont include those bits (ie we have areas of 0 signal)... (ie 1cm^2 must be completely in the large ROI)
                    mean_val = np.mean(values)
                    mean_array[row, col] = mean_val

            sig_max = np.max(mean_array)
            sig_min = np.min(
                mean_array[np.nonzero(mean_array)]
            )  # We initalise the array with 0s but we don't acccept any 0s in the above. Hence we should just ignore 0s here.

            max_loc = np.where(mean_array == sig_max)
            min_loc = np.where(mean_array == sig_min)

            max_loc = (max_loc[0][0], max_loc[1][0])
            min_loc = (min_loc[0][0], min_loc[1][0])

        piu = 100 * (1 - (sig_max - sig_min) / (sig_max + sig_min))

        if self.report:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1)
            fig.set_size_inches(8, 16)
            fig.tight_layout(pad=4)

            theta = np.linspace(0, 2 * np.pi, 360)

            axes[0].imshow(img)
            axes[0].scatter(cxy[0], cxy[1], c="red")
            circle1 = plt.Circle(
                (cxy[0], cxy[1]), mask_unif.radius, color="r", fill=False
            )
            axes[0].add_patch(circle1)
            axes[0].axis("off")
            axes[0].set_title("Centroid Location")

            axes[1].imshow(img)
            axes[1].imshow(lroi, alpha=0.4)
            axes[1].scatter(
                [max_loc[1], min_loc[1]], [max_loc[0], min_loc[0]], c="red", marker="x"
            )

            ROI_min = plt.Rectangle(
                (min_loc[1] - round(5 / res[0]), min_loc[0] - round(5 / res[0])),
                10 / res[0],
                10 / res[0],
                color="y",
                fill=False,
            )
            axes[1].add_patch(ROI_min)
            ROI_max = plt.Rectangle(
                (max_loc[1] - round(5 / res[0]), max_loc[0] - round(5 / res[0])),
                10 / res[0],
                10 / res[0],
                color="y",
                fill=False,
            )
            axes[1].add_patch(ROI_max)

            """axes[1].plot(
                r_small * np.cos(theta) + max_loc[1],
                r_small * np.sin(theta) + max_loc[0],
                c="yellow",
            )"""
            axes[1].annotate(
                "Min = " + str(np.round(sig_min, 1)),
                [min_loc[1], min_loc[0] + 10 / res[0]],
                c="white",
            )

            """axes[1].plot(
                r_small * np.cos(theta) + min_loc[1],
                r_small * np.sin(theta) + min_loc[0],
                c="yellow",
            )"""
            axes[1].annotate(
                "Max = " + str(np.round(sig_max, 1)),
                [max_loc[1], max_loc[0] + 10 / res[0]],
                c="white",
            )

            axes[1].plot(
                r_large * np.cos(theta) + cxy[0],
                r_large * np.sin(theta) + cxy[1] + d_void,
                c="black",
            )

            axes[1].axis("off")
            axes[1].set_title(
                "Percent Integral Uniformity = " + str(np.round(piu, 2)) + "%"
            )

            img_path = os.path.realpath(
                os.path.join(self.report_path, f"{self.img_desc(dcm)}_uniformity.png")
            )
            fig.savefig(img_path, dpi=300)
            plt.close()
            self.report_files.append(img_path)

        return piu, sig_max, sig_min, max_loc, min_loc
