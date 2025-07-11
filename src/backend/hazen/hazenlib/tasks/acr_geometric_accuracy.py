"""
acr_geometric_accuracy

https://www.acraccreditation.org/-/media/acraccreditation/documents/mri/largephantomguidance.pdf

Calculates geometric accuracy for most uniform slice of ACR phantom. The horizontal, vertical and diagonal lengths
are calculated from the mask of the most uniform slice. Rotation matrices around the mask centre are used to determine
diagonal lengths. The results are also visualised.

Created by Yassine Azma (Adapted by Nathan Crossley for local RSCH purposes, 2025)
yassine.azma@rmh.nhs.uk

18/11/2022

"""

import os
import numpy as np
import cv2
import matplotlib.patches as mpatches
from pydicom import Dataset

from src.backend.hazen.hazenlib.HazenTask import HazenTask
from src.backend.hazen.hazenlib.ACRObject import ACRObject
from src.backend.hazen.hazenlib.masking_tools.slice_mask import SliceMask


class ACRGeometricAccuracy(HazenTask):
    """Subclass of HazenTask that contains code relating to calculating
    the geometric accuracy of dcms in the ACR phantom image set.
    """

    def __init__(self, **kwargs):
        # Call initialiser of HazenTask
        super().__init__(**kwargs)

        # Instantiate ACRObject class using dcm list passed in within kwargs.
        self.ACR_obj = ACRObject(self.dcm_list)

    def run(self) -> dict:
        """Entry function for performing geometric accuracy measurement
        using the most uniform image from the ACR phantom image set

        Returns:
            dict: results are returned in a standardised dictionary
                structure specifying the task name, input DICOM Series
                Description + SeriesNumber + InstanceNumber, task
                measurement key-value pairs, optionally path to the
                generated images for visualisation.
        """

        # Identify relevant slice and get dicom and mask
        target_slice = self.ACR_obj.most_uniform_slice
        dcm_geom = self.ACR_obj.dcms[target_slice]
        mask_geom = self.ACR_obj.masks[target_slice]

        # Initialise results dictionary and add image description
        results = self.init_result_dict()
        results["file"] = self.img_desc(dcm_geom)

        try:
            # calculate geometric accuracy using chosen dcm and mask
            lengths_5 = self.get_geometric_accuracy(dcm_geom, mask_geom)

            # append the four calculated lengths to the results dict
            results["measurement"] = {
                "Horizontal distance": round(lengths_5[0], 2),
                "Vertical distance": round(lengths_5[1], 2),
                "Diagonal distance SW": round(lengths_5[2], 2),
                "Diagonal distance SE": round(lengths_5[3], 2),
            }

            # signal to the user that the geomtric accuracy was successfully calculated for this dcm
            print(f"{self.img_desc(dcm_geom)}: Geometric accuracy calculated.")

        except Exception as e:

            # inform the user that the geomtric accuracy could not be calculated and why
            print(
                f"{self.img_desc(dcm_geom)}: Could not calculate geometric accuracy because of: {e}"
            )
            # traceback.print_exc(file=sys.stdout)

        # only return reports if requested
        if self.report:
            results["report_image"] = self.report_files

        return results

    def get_geometric_accuracy(self, dcm: Dataset, mask: SliceMask) -> tuple[float]:
        """Measure geometric accuracy for chosen dcm. This is achieved
        through obtaining an on-axis bounding box for on-axis and rotated
        versions of the mask of the dcm.

        Args:
            dcm (Dataset): Dcm chosen for geometric accuracy task.
            mask (SliceMask): Mask corresponding to chosen dcm.

        Returns:
            tuple[float]: horizontal and vertical distances, as well as diagonals (SW, SE)
        """

        # get dcm "image"
        img = dcm.pixel_array

        # get centre of mask
        cxy = mask.centre

        # get orthognal and diagonal lengths from mask
        length_dict = self.measure_orthogonal_lengths(mask)
        length_dict |= self.measure_diagonal_lengths(mask)

        if self.report:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1)
            fig.set_size_inches(8, 24)
            fig.tight_layout(pad=4)

            # show which dcm has been used and the detected centre
            axes[0].imshow(img)
            axes[0].scatter(cxy[0], cxy[1], c="red")
            axes[0].axis("off")
            axes[0].set_title("Centroid Location")

            # show the elliptical mask used for geometric accuracy calculations
            axes[1].imshow(mask.elliptical_mask)
            axes[1].axis("off")
            axes[1].set_title("Elliptical Mask")

            # on axis 2 show the image and the elliptical mask
            axes[2].imshow(img)
            axes[2].imshow(mask.elliptical_mask, alpha=0.4)

            triplets = (
                ("Horizontal Start", "Horizontal End", "Horizontal Distance"),
                ("Vertical Start", "Vertical End", "Vertical Distance"),
                ("Diagonal Start 1", "Diagonal End 1", "Diagonal Distance 1"),
                ("Diagonal Start 2", "Diagonal End 2", "Diagonal Distance 2"),
            )

            # for each pair of start and end points, plot arrows on axis 2 to symbolise measurements
            for i, (start, end, _) in enumerate(triplets):
                axes[2].annotate(
                    "",
                    xy=length_dict[end],
                    xytext=length_dict[start],
                    arrowprops=dict(
                        arrowstyle="->", color=f"C{i}", lw=1, mutation_scale=15
                    ),
                )

            # add legend handles with colors matching those of arrows and labels corresponding to measured geometric accuracy values
            legend_handles = [
                mpatches.Patch(
                    color=f"C{i}", label=f"{np.round(length_dict[dist], 2)} mm"
                )
                for i, (_, _, dist) in enumerate(triplets)
            ]

            # configure axis 2
            axes[2].legend(handles=legend_handles, loc="upper right", title="Distances")
            axes[2].axis("off")
            axes[2].set_title("Geometric Accuracy for Slice 5")

            # Work out path to save plots at
            img_path = os.path.realpath(
                os.path.join(
                    self.report_path, f"{self.img_desc(dcm)}_geom_accuracy.png"
                )
            )

            # save plots and save paths
            fig.savefig(img_path, dpi=300)
            self.report_files.append(img_path)
            plt.close()

        return (
            length_dict["Horizontal Distance"],
            length_dict["Vertical Distance"],
            length_dict["Diagonal Distance 1"],
            length_dict["Diagonal Distance 2"],
        )

    def measure_orthogonal_lengths(self, mask: SliceMask) -> dict:
        """Measures orthogonal lengths for a given dcm mask.
        This is achieved by taking an on-axis bounding box about
        the mask's elliptical mask.

        Args:
            mask (SliceMask): Dcm mask chosen for geometric accuracy calculations.

        Returns:
            dict: Dictionary containing horizontal and vertical start points, end points and distances.
        """

        # work out pixel spacing in mm for x and y
        dx, dy = self.ACR_obj.pixel_spacing

        # get on axis bounding box around mask
        x, y, w, h = cv2.boundingRect(mask.elliptical_mask)

        # calculate start point, end point and distance for horizontal orientation
        horizontal_start = (x, mask.centre[1])
        horizontal_end = (x + w, mask.centre[1])
        horizontal_distance = w * dx

        # calculate start point, end point and distance for vertical orientation
        vertical_start = (mask.centre[0], y)
        vertical_end = (mask.centre[0], y + h)
        vertical_distance = h * dy

        # construct results dict
        length_dict = {
            "Horizontal Start": horizontal_start,
            "Horizontal End": horizontal_end,
            "Horizontal Distance": horizontal_distance,
            "Vertical Start": vertical_start,
            "Vertical End": vertical_end,
            "Vertical Distance": vertical_distance,
        }

        return length_dict

    def measure_diagonal_lengths(self, mask: SliceMask) -> dict:
        """Measures diagonal lengths for a given dcm mask.
        This is achieved by rotating the mask by 45 degrees
        and repeating the on-axis analysis.

        Args:
            mask (SliceMask): Mask to use for geometric accuracy calculations

        Returns:
            dict: Dictionary containing the two diagonal start points, end points and distances.
        """

        # rotate mask by 45 degrees to measure diagonal distances
        mask_45 = mask.get_rotated_mask(45)

        # measured orthogonal distances of rotated mask (i.e. diagonal distances)
        length_dict_diag = self.measure_orthogonal_lengths(mask_45)

        # adjust keys of new dict to reflect diagonal distances
        # transform points back to un-rotated reference frame before assigning to keys

        key_pairs = (
            ("Diagonal Start 1", "Horizontal Start"),
            ("Diagonal End 1", "Horizontal End"),
            ("Diagonal Distance 1", "Horizontal Distance"),
            ("Diagonal Start 2", "Vertical Start"),
            ("Diagonal End 2", "Vertical End"),
            ("Diagonal Distance 2", "Vertical Distance"),
        )

        for new_key, old_key in key_pairs:
            val = length_dict_diag.pop(old_key)
            length_dict_diag[new_key] = (
                val
                if isinstance(val, (int, float))
                else mask_45.transform_point_to_orig_frame(val)
            )

        return length_dict_diag
