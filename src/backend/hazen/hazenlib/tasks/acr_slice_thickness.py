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

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pydicom.dataset import FileDataset

from backend.hazen.hazenlib.HazenTask import HazenTask
from backend.hazen.hazenlib.ACRObject import ACRObject
from backend.hazen.hazenlib.masking_tools.slice_mask import SliceMask
from backend.hazen.hazenlib.masking_tools.contour_validation import (
    is_slice_thickness_insert,
)
from backend.hazen.hazenlib.utils import Point, Line
from backend.hazen.hazenlib.tasks.support_classes.line_slice_thickness import (
    LineSliceThickness,
)


class ACRSliceThickness(HazenTask):
    """Slice width measurement class for DICOM images of the ACR phantom."""

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
            LineSliceThickness(sublines[0].start, sublines[1].start),
            LineSliceThickness(sublines[0].end, sublines[1].end),
        ]

        # Final lines are sublines of connecting lines
        final_lines = [line.get_subline(perc=95) for line in connecting_lines]
        for line in final_lines:
            line.get_signal(image)

        return final_lines
