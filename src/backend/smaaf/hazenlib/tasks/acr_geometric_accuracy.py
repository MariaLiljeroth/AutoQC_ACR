"""
ACR Geometric Accuracy

https://www.acraccreditation.org/-/media/acraccreditation/documents/mri/largephantomguidance.pdf

Calculates geometric accuracy for slices 1 and 5 of the ACR phantom.

This script calculates the horizontal and vertical lengths of the ACR phantom in Slice 1 in accordance with the ACR Guidance.
This script calculates the horizontal, vertical and diagonal lengths of the ACR phantom in Slice 5 in accordance with the ACR Guidance.
The average distance measurement error, maximum distance measurement error and coefficient of variation of all distance
measurements is reported as recommended by IPEM Report 112, "Quality Control and Artefacts in Magnetic Resonance Imaging".

This is done by first producing a binary mask for each respective slice. Line profiles are drawn with aid of rotation
matrices around the centre of the test object to determine each respective length. The results are also visualised.

Created by Yassine Azma
yassine.azma@rmh.nhs.uk

18/11/2022
"""

import os
import numpy as np
import cv2
import matplotlib.patches as mpatches
from pydicom.dataset import FileDataset

from hazenlib.HazenTask import HazenTask
from hazenlib.ACRObject import ACRObject
from hazenlib.slice_mask import SliceMask


class ACRGeometricAccuracy(HazenTask):
    """Geometric accuracy measurement class for DICOM images of the ACR phantom

    Inherits from HazenTask class
    """

    def __init__(self, **kwargs):
        # Run initialiser of HazenTask
        super().__init__(**kwargs)

        # Instantiate ACRObject class.
        self.ACR_obj = ACRObject(self.dcm_list)

    def run(self) -> dict:
        """Main function for performing geometric accuracy measurement
        using the fifth slice from the ACR phantom image set

        Returns:
            dict: results are returned in a standardised dictionary structure specifying the task name, input DICOM Series Description + SeriesNumber + InstanceNumber, task measurement key-value pairs, optionally path to the generated images for visualisation
        """

        # Identify relevant slice and get dicom and mask
        target_slice = self.ACR_obj.most_uniform_slice
        dcm_geom = self.ACR_obj.dcms[target_slice]
        mask_geom = self.ACR_obj.masks[target_slice]

        # Initialise results dictionary
        results = self.init_result_dict()
        results["file"] = self.img_desc(dcm_geom)

        try:
            lengths_5 = self.get_geometric_accuracy_slice5(dcm_geom, mask_geom)
            results["measurement"] = {
                "Horizontal distance": round(lengths_5[0], 2),
                "Vertical distance": round(lengths_5[1], 2),
                "Diagonal distance SW": round(lengths_5[2], 2),
                "Diagonal distance SE": round(lengths_5[3], 2),
            }
            print(f"{self.img_desc(dcm_geom)}: Geometric accuracy calculated.")

        except Exception as e:
            print(
                f"{self.img_desc(dcm_geom)}: Could not calculate geometric accuracy because of: {e}"
            )
            # traceback.print_exc(file=sys.stdout)

        # only return reports if requested
        if self.report:
            results["report_image"] = self.report_files

        return results

    def get_geometric_accuracy_slice5(
        self, dcm: FileDataset, mask: SliceMask
    ) -> tuple[float]:
        """Measure geometric accuracy for slice 5

        Args:
            dcm (FileDataset): DICOM image object
            mask (SliceMask): Mask corresponding to chosen DICOM

        Returns:
            tuple[float]: horizontal and vertical distances, as well as diagonals (SW, SE)
        """
        img = dcm.pixel_array
        cxy = mask.centre

        # get orthognal and diagonal lengths from mask
        length_dict = self.measure_orthogonal_lengths(mask)
        length_dict |= self.measure_diagonal_lengths(mask)

        if self.report:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1)
            fig.set_size_inches(8, 24)
            fig.tight_layout(pad=4)

            axes[0].imshow(img)
            axes[0].scatter(cxy[0], cxy[1], c="red")
            axes[0].axis("off")
            axes[0].set_title("Centroid Location")

            axes[1].imshow(mask.elliptical_mask)
            axes[1].axis("off")
            axes[1].set_title("Elliptical Mask")

            axes[2].imshow(img)
            axes[2].imshow(mask.elliptical_mask, alpha=0.4)

            triplets = (
                ("Horizontal Start", "Horizontal End", "Horizontal Distance"),
                ("Vertical Start", "Vertical End", "Vertical Distance"),
                ("Diagonal Start 1", "Diagonal End 1", "Diagonal Distance 1"),
                ("Diagonal Start 2", "Diagonal End 2", "Diagonal Distance 2"),
            )

            for i, (start, end, _) in enumerate(triplets):
                axes[2].annotate(
                    "",
                    xy=length_dict[end],
                    xytext=length_dict[start],
                    arrowprops=dict(
                        arrowstyle="->", color=f"C{i}", lw=1, mutation_scale=15
                    ),
                )

            legend_handles = [
                mpatches.Patch(
                    color=f"C{i}", label=f"{np.round(length_dict[dist], 2)} mm"
                )
                for i, (_, _, dist) in enumerate(triplets)
            ]

            axes[2].legend(handles=legend_handles, loc="upper right", title="Distances")

            axes[2].axis("off")
            axes[2].set_title("Geometric Accuracy for Slice 5")

            img_path = os.path.realpath(
                os.path.join(
                    self.report_path, f"{self.img_desc(dcm)}_geom_accuracy.png"
                )
            )
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
        """Measures orthogonal lengths for a mask.

        Args:
            mask (SliceMask): Mask to find orthogonal lengths of.

        Returns:
            dict: Dictionary containing horizontal and vertical start points, end points and distances.
        """
        dx, dy = self.ACR_obj.pixel_spacing

        # get on axis bounding box around mask
        x, y, w, h = cv2.boundingRect(mask.elliptical_mask)

        # calculate start point, end point and distance for horizontal and vertical orientations
        horizontal_start = (x, mask.centre[1])
        horizontal_end = (x + w, mask.centre[1])
        horizontal_distance = w * dx

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
        """Measures diagonal lengths for a mask.

        Args:
            mask (SliceMask): Mask to find diagonal lengths of.

        Returns:
            dict: Dictionary containing the two diagonal start points, end points and distances.
        """

        # rotate mask by 45 degrees to measure diagonal distances
        mask_45 = mask.get_rotated_mask(45)

        # measured orthogonal distances of rotated mask (i.e. diagonal distances)
        length_dict_diag = self.measure_orthogonal_lengths(mask_45)
        key_pairs = (
            ("Diagonal Start 1", "Horizontal Start"),
            ("Diagonal End 1", "Horizontal End"),
            ("Diagonal Distance 1", "Horizontal Distance"),
            ("Diagonal Start 2", "Vertical Start"),
            ("Diagonal End 2", "Vertical End"),
            ("Diagonal Distance 2", "Vertical Distance"),
        )

        # adjust keys of new dict to reflect diagonal distances
        # transform points back to un-rotated reference frame before assigning to keys
        for new_key, old_key in key_pairs:
            val = length_dict_diag.pop(old_key)
            length_dict_diag[new_key] = (
                val
                if isinstance(val, (int, float))
                else mask_45.transform_point_to_orig_frame(val)
            )

        return length_dict_diag

    # @staticmethod
    # def distortion_metric(L):
    #     """Calculate the distortion metric based on length

    #     Args:
    #         L (tuple): horizontal and vertical distances from slices 1 and 5

    #     Returns:
    #         tuple of floats: mean_err, max_err, cov_l
    #     """
    #     err = [x - 190 for x in L]
    #     mean_err = np.mean(err)

    #     max_err = np.max(np.absolute(err))
    #     cov_l = 100 * np.std(L) / np.mean(L)

    #     return mean_err, max_err, cov_l

    # @staticmethod
    # def distortion_metric_MedPhantom(L):
    #     """Calculate the distortion metric based on length

    #     Args:
    #         L (tuple): horizontal and vertical distances from slices 1 and 5

    #     Returns:
    #         tuple of floats: mean_err, max_err, cov_l
    #     """
    #     err = [x - 165 for x in L]
    #     mean_err = np.mean(err)

    #     max_err = np.max(np.absolute(err))
    #     cov_l = 100 * np.std(L) / np.mean(L)

    #     return mean_err, max_err, cov_l
