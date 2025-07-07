"""
acr_slice_thickness.py

Calculates the slice thickness from the first slice of the ACR phantom. The slice thickness insert is located
and two ramps placed within it. The FWHM values of each ramp are determined and the formula described in the ACR
guidance used to calculate the slice thickness.

Created by Yassine Azma (Adapted by Nathan Crossley for local RSCH purposes, 2025)
yassine.azma@rmh.nhs.uk

31/01/2022

"""

import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pydicom import Dataset

from backend.hazen.hazenlib.HazenTask import HazenTask
from backend.hazen.hazenlib.ACRObject import ACRObject
from backend.hazen.hazenlib.masking_tools.slice_mask import SliceMask
from backend.hazen.hazenlib.masking_tools.contour_validation import (
    is_slice_thickness_insert,
)
from backend.hazen.hazenlib.tasks.support_classes.point_2d import Point2D
from backend.hazen.hazenlib.tasks.support_classes.line_2d import Line2D
from backend.hazen.hazenlib.tasks.support_classes.line_2d_slice_thickness import (
    Line2DSliceThickness,
)


class ACRSliceThickness(HazenTask):
    """Subclass of HazenTask that contains code relating to calculating
    the slice thickness of dcm images of the ACR phantom image set.
    """

    def __init__(self, **kwargs):
        # Call initialiser of HazenTask
        super().__init__(**kwargs)

        # Instantiate ACRObject class using dcm list passed in within kwargs
        self.ACR_obj = ACRObject(self.dcm_list)

    def run(self) -> dict:
        """Entrypoint function to trigger the slice thickness calculation,
        using first slice of the ACR phantom image set.

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

        # Initialise results dictionary and add image description
        results = self.init_result_dict()
        results["file"] = self.img_desc(dcm_slice_th)

        try:
            # get slice thickness of chosen dcm and mask
            result = self.get_slice_thickness(dcm_slice_th, mask)

            # append results to the results dict
            results["measurement"] = {"slice width mm": round(result, 2)}

            # signal to user that slice thickness has been calculated for given dcm
            print(f"{self.img_desc(dcm_slice_th)}: Slice thickness calculated.")

        except Exception as e:

            # alert the user that slice thickness could not be calculated and why
            print(
                f"{self.img_desc(dcm_slice_th)}: Could not calculate slice thickness because of: {e}"
            )
            # traceback.print_exc(file=sys.stdout)

        # only return reports if requested
        if self.report:
            results["report_image"] = self.report_files

        return results

    def get_slice_thickness(self, dcm: Dataset, mask: SliceMask) -> float:
        """Measure slice thickness of the provided dcm.
        Identify the slice thickness insert, measure the signal across two profile lines,
        measure the FWHMs and use these to calculate the slice thickness as per ACR guidance.

        Args:
            dcm (Dataset): Dcm of chosen ACR slice for slice thickness task.
            mask (SliceMask): Corresponding mask of chosen dcm.

        Returns:
            float: Measured slice thickness.
        """

        # get 'image' of dcm
        image = dcm.pixel_array

        # define interpolation factor for upscaling image (for accuracy when positioning line profiles)
        interp_factor = 4

        # define pixel spacing in mm using interp_factor
        interp_pixel_mm = [dist / interp_factor for dist in self.ACR_obj.pixel_spacing]

        # resize image and mask for increased accuracy in line profile placement.
        new_dims = tuple([interp_factor * dim for dim in image.shape])
        image = cv2.resize(image, new_dims, interpolation=cv2.INTER_CUBIC)
        mask = mask.get_scaled_mask(interp_factor)

        # place profile lines on the image within the slice thickness insert.
        lines = self.place_lines(image, mask)

        # for each profile line, get the FWHM and convert to mm.
        for line in lines:
            line.get_FWHM()
            line.FWHM *= np.mean(interp_pixel_mm)

        # calculate slice thickness as per NEMA standard.
        slice_thickness = (
            0.2 * (lines[0].FWHM * lines[1].FWHM) / (lines[0].FWHM + lines[1].FWHM)
        )

        # report images if requested
        if self.report:
            fig, axes = plt.subplots(1, 3, figsize=(16, 8))

            # show sliced used for calculations
            axes[0].imshow(image)

            # process one line profile at a time
            for i, line in enumerate(lines):

                # overlay lines on image to show line positioning
                axes[0].plot([line.start.x, line.end.x], [line.start.y, line.end.y])

                # plot raw signal, with increased transparency
                axes[i + 1].plot(
                    line.signal.x,
                    line.signal.y,
                    label="Raw signal",
                    alpha=0.25,
                    color=f"C{i}",
                )

                # plot fitted signal to show accuracy of the model
                axes[i + 1].plot(
                    line.fitted.x,
                    line.fitted.y,
                    label="Fitted piecewise sigmoid",
                    color=f"C{i}",
                )

                # display legend
                axes[i + 1].legend(loc="lower right", bbox_to_anchor=(1, -0.2))

            # configure plot settings for axis 0.
            axes[0].axis("off")
            axes[0].set_title("Plot showing placement of profile lines.")

            # configure plot settings for axis 1.
            axes[1].set_title("Pixel profile across blue line.")
            axes[1].set_xlabel("Distance along blue line (pixels)")
            axes[1].set_ylabel("Pixel value")

            # Configure plot settings for axis 2.
            axes[2].set_title("Pixel profile across orange line.")
            axes[2].set_xlabel("Distance along orange line (pixels)")
            axes[2].set_ylabel("Pixel value")
            plt.tight_layout()

            # Work out where to save the plot.
            image_path = os.path.realpath(
                os.path.join(
                    self.report_path, f"{self.img_desc(dcm)}_slice_thickness.png"
                )
            )

            # Save the plot and save path to report files list.
            fig.savefig(image_path, dpi=300)
            plt.close()
            self.report_files.append(image_path)

        return slice_thickness

    def place_lines(self, image: np.ndarray, mask: np.ndarray) -> list[Line2D]:
        """Places two profile lines on image within slice thickness insert.
        Works even for a rotated phantom.

        Args:
            image (np.ndarray): Pixel array from slice thickness dcm.
            mask (SliceMask): mask for slice thickness task.

        Returns:
            final_lines (list[Line2D]): A list of the two lines as Line2D objects.
        """

        # Get contour of slice thickness insert.
        insert = [c for c in mask.contours if is_slice_thickness_insert(c, mask.shape)][
            0
        ]

        # Create list of Point2D objects for the four corners of the contour
        corners = cv2.boxPoints(cv2.minAreaRect(insert))
        corners = [Point2D(*p) for p in corners]

        # Define short sides of contours by list of Line2D objects
        corners = sorted(corners, key=lambda point: corners[0].get_distance_to(point))
        short_sides = [Line2D(*corners[:2]), Line2D(*corners[2:])]

        # Get sublines of short sides to inset line endpoints away from insert long sides.
        sublines = [line.get_subline(percentage=30) for line in short_sides]

        # Force start point to be higher in y by swapping points if necessary.
        for line in sublines:
            if line.start.y < line.end.y:
                line.point_swap()

        # Define connecting lines between the sublines of the short sides.
        connecting_lines = [
            Line2DSliceThickness(sublines[0].start, sublines[1].start),
            Line2DSliceThickness(sublines[0].end, sublines[1].end),
        ]

        # Final lines are sublines of connecting lines to inset lines away from short edges of insert.
        final_lines = [line.get_subline(percentage=95) for line in connecting_lines]

        # get signal across each of the final line objects.
        for line in final_lines:
            line.get_signal(image)

        return final_lines
