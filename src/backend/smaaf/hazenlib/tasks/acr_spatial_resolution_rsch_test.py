"""
ACR Spatial Resolution (MTF)

https://www.acraccreditation.org/-/media/acraccreditation/documents/mri/largephantomguidance.pdf

Calculates the effective resolution (MTF50) for slice 1 for the ACR phantom. This is done in accordance with the
methodology described in Section 3 of the following paper:

https://opg.optica.org/oe/fulltext.cfm?uri=oe-22-5-6040&id=281325

WARNING: The phantom must be slanted for valid results to be produced. This test is not within the scope of ACR
guidance.

This script first identifies the rotation angle of the ACR phantom using slice 1. It provides a warning if the
slanted angle is less than 3 degrees.

The location of the ramps within the slice thickness are identified and a square ROI is selected around the anterior
edge of the slice thickness insert.

A rudimentary edge response function is generated based on the edge within the ROI to provide initialisation values for
the 2D normal cumulative distribution fit of the ROI.

The edge is then super-sampled in the direction of the bright-dark transition of the edge and binned at right angles
based on the edge slope determined from the 2D Normal CDF fit of the ROI to obtain the edge response function.

This super-sampled ERF is then fitted using a weighted sigmoid function. The raw data and this fit are then used to
determine the LSF and the subsequent MTF. The MTF50 for both raw and fitted data are reported.

The results are also visualised.

Created by Yassine Azma
yassine.azma@rmh.nhs.uk

22/02/2023
"""

import os
import sys
import traceback
import numpy as np

import cv2
import scipy
import skimage.morphology
import skimage.measure

from hazenlib.HazenTask import HazenTask
from hazenlib.ACRObject import ACRObject
from hazenlib.logger import logger

import sys
import matplotlib.pyplot as plt
import scipy.ndimage
import skimage.segmentation
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import math
from enum import Enum


class ResOptions(Enum):
    DotMatrixMethod = 1
    MTFMethod = 2
    ContrastResponseMethod = 3
    Manual = 4


class ACRSpatialResolution(HazenTask):
    """Spatial resolution measurement class for DICOM images of the ACR phantom

    Inherits from HazenTask class
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ACR_obj = ACRObject(self.dcm_list, kwargs)
        self.ResOption = ResOptions.MTFMethod

    def run(self) -> dict:
        """Main function for performing spatial resolution measurement
        using slice 1 from the ACR phantom image set

        Returns:
            dict: results are returned in a standardised dictionary structure specifying the task name, input DICOM Series Description + SeriesNumber + InstanceNumber, task measurement key-value pairs, optionally path to the generated images for visualisation
        """

        # Identify relevant slices
        mtf_dcm = self.ACR_obj.dcms[0]
        
        # Initialise results dictionary
        results = self.init_result_dict()
        results["file"] = self.img_desc(mtf_dcm)

        try:
            if self.ResOption == ResOptions.MTFMethod:
                raw_res, fitted_res = self.get_mtf50(mtf_dcm)
                results["measurement"] = {
                    "raw mtf50": round(raw_res, 2),
                    "fitted mtf50": round(fitted_res, 2),
                }

            else:
                raise Exception("Unexpected option in spatial res module.")

        except Exception as e:
            print(
                f"Could not calculate the spatial resolution for {self.img_desc(mtf_dcm)} because of : {e}"
            )
            traceback.print_exc(file=sys.stdout)
            raise Exception(e)

        # only return reports if requested
        if self.report:
            results["report_image"] = self.report_files

        return results


    def get_mtf50(self, dcm):
        """_summary_

        Args:
            dcm (pydicom.Dataset): DICOM image object

        Returns:
            tuple: _description_
        """
        img = dcm.pixel_array
        cxy = self.ACR_obj.centre
        
        # Rotate image to be at best angle for mtf
        phantomRot = self.ACR_obj.rot_angle
        bestTheta = 3.5 * np.sign(phantomRot)
        dTheta = bestTheta - phantomRot
        img = skimage.transform.rotate(img, dTheta, resize=False, center = cxy, preserve_range=True)
        
        if "PixelSpacing" in dcm:
            res = dcm.PixelSpacing  # In-plane resolution from metadata
        else:
            import hazenlib.utils

            res = hazenlib.utils.GetDicomTag(dcm, (0x28, 0x30))

        ramp_insert = self.ACR_obj.find_ramps_insert(img)
        
        ramp_x = int(cxy[0])
        ramp_y = self.y_position_for_ramp(res, img, cxy)

        width = int(13 * img.shape[0] / 256)
        crop_img = self.crop_image(img, ramp_x, ramp_y, width)
        # HAPPY THAT IT WORKS UP UNTIL HERE
        
        
        edge_type, direction = self.get_edge_type(crop_img)
        slope, surface = self.fit_normcdf_surface(crop_img, edge_type, direction)
        erf = self.sample_erf(crop_img, slope, edge_type)
        erf_fit = self.fit_erf(erf)

        freq, lsf_raw, MTF_raw = self.calculate_MTF(erf, res)
        _, lsf_fit, MTF_fit = self.calculate_MTF(erf_fit, res)

        eff_raw_res = self.identify_MTF50(freq, MTF_raw)
        eff_fit_res = self.identify_MTF50(freq, MTF_fit)

        if self.report:
            edge_loc = self.edge_location_for_plot(crop_img, edge_type)

            fig, axes = plt.subplots(5, 1)
            fig.set_size_inches(8, 40)
            fig.tight_layout(pad=4)

            axes[0].imshow(img, interpolation="none")
            rect = patches.Rectangle(
                (ramp_x - width // 2 - 1, ramp_y - width // 2 - 1),
                width,
                width,
                linewidth=1,
                edgecolor="w",
                facecolor="none",
            )
            axes[0].add_patch(rect)
            # axes[0].axis("off")
            axes[0].set_title("Segmented Edge")

            axes[1].imshow(crop_img)
            if edge_type == "vertical":
                axes[1].plot(
                    np.arange(0, width - 1),
                    np.mean(edge_loc) - slope * np.arange(0, width - 1),
                    color="r",
                )
            else:
                axes[1].plot(
                    np.mean(edge_loc) + slope * np.arange(0, width - 1),
                    np.arange(0, width - 1),
                    color="r",
                )

            axes[1].axis("off")
            axes[1].set_title("Cropped Edge", fontsize=14)

            axes[2].plot(erf, "rx", ms=5, label="Raw Data")
            axes[2].plot(erf_fit, "k", lw=3, label="Fitted Data")
            axes[2].set_ylabel("Signal Intensity")
            axes[2].set_xlabel("Pixel")
            axes[2].grid()
            axes[2].legend(fancybox="true")
            axes[2].set_title("ERF", fontsize=14)

            axes[3].plot(lsf_raw, "rx", ms=5, label="Raw Data")
            axes[3].plot(lsf_fit, "k", lw=3, label="Fitted Data")
            axes[3].set_ylabel(r"$\Delta$" + " Signal Intensity")
            axes[3].set_xlabel("Pixel")
            axes[3].grid()
            axes[3].legend(fancybox="true")
            axes[3].set_title("LSF", fontsize=14)

            axes[4].plot(
                freq,
                MTF_raw,
                "rx",
                ms=8,
                label=f"Raw Data - {round(eff_raw_res, 2)}mm @ 50%",
            )
            axes[4].plot(
                freq,
                MTF_fit,
                "k",
                lw=3,
                label=f"Weighted Sigmoid Fit of ERF - {round(eff_fit_res, 2)}mm @ 50%",
            )
            axes[4].set_xlabel("Spatial Frequency (lp/mm)")
            axes[4].set_ylabel("Modulation Transfer Ratio")
            axes[4].set_xlim([-0.05, 1])
            axes[4].set_ylim([0, 1.05])
            axes[4].grid()
            axes[4].legend(fancybox="true")
            axes[4].set_title("MTF", fontsize=14)

            img_path = os.path.realpath(
                os.path.join(self.report_path, f"{self.img_desc(dcm)}_MTF.png")
            )
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return eff_raw_res, eff_fit_res
