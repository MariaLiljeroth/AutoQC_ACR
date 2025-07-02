import os

import matplotlib.pyplot as plt
import numpy as np
import cv2

from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
from scipy.special import sici

from backend.smaaf.hazenlib.HazenTask import HazenTask
from backend.smaaf.hazenlib.ACRObject import ACRObject
from backend.smaaf.hazenlib.image_processing_tools.contour_validation import (
    is_slice_thickness_insert,
)


class ACRSpatialResolution(HazenTask):
    """Spatial resolution measurement class for DICOM images of the ACR phantom

    Inherits from HazenTask class
    """

    TARGET_THETA_INSERT = 3
    SIZE_ROI = 10

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ACR_obj = ACRObject(self.dcm_list, kwargs)

    def run(self):
        target_slice = 0
        mtf_dcm = self.ACR_obj.dcms[target_slice]
        mask = self.ACR_obj.masks[target_slice]

        results = self.init_result_dict()
        results["file"] = self.img_desc(mtf_dcm)

        try:
            mtf50 = self.get_mtf50(mtf_dcm, mask)
            results["measurement"] = {"mtf50": mtf50}
            print(f"{self.img_desc(mtf_dcm)}: Spatial resolution calculated.")

        except Exception as e:
            print(
                f"Could not calculate the spatial resolution for {self.img_desc(mtf_dcm)} because of : {e}"
            )

        if self.report:
            results["report_image"] = self.report_files

        return results

    def get_mtf50(self, dcm, mask):
        self.image_orig = dcm.pixel_array
        self.roi, self.roi_centre, self.roi_bounds, self.image_rotated = self.get_roi(
            mask
        )

        self.esf = self.get_esf_raw()
        self.esf_fitted = self.get_esf_fitted()

        self.lsf = self.get_lsf(self.esf)
        self.lsf_fitted = self.get_lsf(self.esf_fitted)

        self.mtf = self.get_mtf(self.lsf)
        self.mtf_fitted = self.get_mtf(self.lsf_fitted)

        def simple_interpolate(target_y: float, y: list, x: list) -> float:
            crossing_index = np.where(y < target_y)[0][0]
            x1, x2 = x[crossing_index - 1], x[crossing_index]
            y1, y2 = y[crossing_index - 1], y[crossing_index]
            targetX = x1 + (target_y - y1) * (x2 - x1) / (y2 - y1)
            return targetX

        mtf50 = simple_interpolate(0.5, self.mtf_fitted[1], self.mtf_fitted[0])
        mtf05 = simple_interpolate(0.005, self.mtf_fitted[1], self.mtf_fitted[0])

        if self.report:
            fig, axes = plt.subplots(4, 1, figsize=(8, 16))

            DISPLAY_PAD = 20
            x1_roi, x2_roi, y1_roi, y2_roi = self.roi_bounds
            x1_display_frame = x1_roi - DISPLAY_PAD
            x2_display_frame = x2_roi + DISPLAY_PAD
            y1_display_frame = y1_roi - DISPLAY_PAD
            y2_display_frame = y2_roi + DISPLAY_PAD

            x1_roi_in_display = DISPLAY_PAD
            x2_roi_in_display = x2_display_frame - x1_display_frame - DISPLAY_PAD
            y1_roi_in_display = DISPLAY_PAD
            y2_roi_in_display = y2_display_frame - y1_display_frame - DISPLAY_PAD

            axes[0].imshow(
                self.image_rotated[
                    y1_display_frame:y2_display_frame, x1_display_frame:x2_display_frame
                ],
            )
            axes[0].vlines(
                [x1_roi_in_display, x2_roi_in_display],
                y1_roi_in_display,
                y2_roi_in_display,
            )
            axes[0].hlines(
                [y1_roi_in_display, y2_roi_in_display],
                x1_roi_in_display,
                x2_roi_in_display,
            )
            axes[1].scatter(self.esf[0], self.esf[1], label="ESF")
            axes[1].plot(self.esf_fitted[0], self.esf_fitted[1], label="ESF fitted")
            axes[1].set_xlabel("Perpendicular distance from edge (mm)")
            axes[1].set_ylabel("Pixel value")
            axes[1].legend()

            # axes[2].scatter(self.lsf[0], self.lsf[1], label="LSF")
            axes[2].plot(self.lsf_fitted[0], self.lsf_fitted[1], label="LSF fitted")
            axes[2].set_xlabel("Perpendicular distance from edge (mm)")
            axes[2].set_ylabel("Pixel value gradient")
            axes[2].legend()

            # axes[3].scatter(self.mtf[0], self.mtf[1], label="MTF")
            axes[3].plot(self.mtf_fitted[0], self.mtf_fitted[1], label="MTF fitted")
            axes[3].set_xlabel("Spatial frequency (mm^-1)")
            axes[3].set_ylabel("MTF")
            axes[3].set_xlim(0, mtf05)
            axes[3].legend()

            image_path = os.path.realpath(
                os.path.join(
                    self.report_path,
                    f"{self.img_desc(dcm)}_spatial_resolution.png",
                )
            )

            fig.tight_layout()
            fig.savefig(image_path, dpi=300)
            plt.close()
            self.report_files.append(image_path)

        return mtf50

    def get_roi(self, mask):
        def get_insert_contour(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            insert = [c for c in contours if is_slice_thickness_insert(c, mask.shape)][
                0
            ]
            return insert

        insert = get_insert_contour(mask)
        image_rotated = self.rotate_rel_to_insert(self.image_orig, insert)
        mask_rotated = self.rotate_rel_to_insert(mask, insert)

        insert_rotated = get_insert_contour(mask_rotated)
        roi_centre = self.define_ROI_centre(insert_rotated)

        x1 = int(roi_centre[0] - self.SIZE_ROI // 2)
        x2 = int(roi_centre[0] + self.SIZE_ROI // 2)
        y1 = int(roi_centre[1] - self.SIZE_ROI // 2)
        y2 = int(roi_centre[1] + self.SIZE_ROI // 2)

        roi = image_rotated[y1:y2, x1:x2]

        return roi, roi_centre, (x1, x2, y1, y2), image_rotated

    @classmethod
    def rotate_rel_to_insert(cls, image, insert) -> np.ndarray:
        _, (w, h), theta = cv2.minAreaRect(insert)
        if w < h:
            theta = theta - 90

        theta_to_apply = theta - cls.TARGET_THETA_INSERT

        (image_h, image_w) = image.shape[:2]
        # Calculate the center of the image
        center = (image_w // 2, image_h // 2)

        # Calculate the rotation matrix
        matrix = cv2.getRotationMatrix2D(center, theta_to_apply, 1.0)

        # Get the new bounding box dimensions after rotation
        abs_cos = abs(matrix[0, 0])
        abs_sin = abs(matrix[0, 1])

        # Calculate the new width and height
        new_w = int(image_h * abs_sin + image_w * abs_cos)
        new_h = int(image_h * abs_cos + image_w * abs_sin)

        # Adjust the rotation matrix to account for translation (shifting the image to prevent clipping)
        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]

        # Rotate the image and resize it to the new size (without clipping)
        image_rotated = cv2.warpAffine(
            image,
            matrix,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return image_rotated

    @staticmethod
    def define_ROI_centre(insert_rotated):
        corners = np.intp(cv2.boxPoints(cv2.minAreaRect(insert_rotated)))
        test_point = corners[0]
        centre = np.intp(
            np.mean(
                (
                    test_point,
                    sorted(
                        corners[1:],
                        key=lambda c: np.sqrt(np.sum((c - test_point) ** 2)),
                    )[1],
                ),
                axis=0,
            )
        )
        return centre

    def get_esf_raw(self):
        theta_rad = np.deg2rad(self.TARGET_THETA_INSERT)

        normal_x = -np.sin(theta_rad)
        normal_y = np.cos(theta_rad)

        y, x = np.indices(self.roi.shape)
        dx = x - self.roi_centre[0]
        dy = y - self.roi_centre[1]

        perp_distances = (dx * normal_x + dy * normal_y).flatten()
        pix_vals = self.roi.flatten()

        sorted_idxs = np.argsort(perp_distances)
        perp_distances = perp_distances[sorted_idxs] * np.mean(
            self.ACR_obj.pixel_spacing
        )
        pix_vals = pix_vals[sorted_idxs]
        esf = np.array([perp_distances, pix_vals])
        return esf

    def get_esf_fitted(self):
        def esf_func(x, c_1, c_2, alpha, m):
            return c_1 / np.pi * sici(alpha * np.pi * (x - m))[0] + c_1 / 2 + c_2

        popt, _ = curve_fit(
            esf_func,
            self.esf[0],
            self.esf[1],
            p0=[np.ptp(self.esf[1]), np.min(self.esf[1]), 1, np.median(self.esf[0])],
        )

        x_range = np.linspace(self.esf[0][0], self.esf[0][-1], 1000)
        esf_fitted = np.array(
            [
                x_range,
                esf_func(x_range, *popt),
            ]
        )
        return esf_fitted

    def get_lsf(self, esf):
        return np.vstack([esf[0], np.gradient(esf[1], esf[0])])

    def get_mtf(self, lsf):
        dx = np.mean(np.diff(lsf[0]))
        mtf = np.abs(fft(lsf[1]))
        mtf /= mtf[0]

        freqs = fftfreq(len(lsf[1]), dx)
        pos = freqs > 0
        mtf = np.vstack([freqs[pos], mtf[pos]])

        return mtf


# @classmethod
# def get_res_matrix(cls, image):
#     contour = cls.locate_insert(image)
#     res_matrix = cls.crop_within_contour_axis_aligned(image, contour)
#     return res_matrix

# @staticmethod
# def crop_within_contour_axis_aligned(image: np.ndarray, contour) -> np.ndarray:

#     _, _, theta = cv2.minAreaRect(contour)
#     (h, w) = image.shape[:2]
#     # Calculate the center of the image
#     center = (w // 2, h // 2)

#     # Calculate the rotation matrix
#     matrix = cv2.getRotationMatrix2D(center, theta, 1.0)

#     # Get the new bounding box dimensions after rotation
#     abs_cos = abs(matrix[0, 0])
#     abs_sin = abs(matrix[0, 1])

#     # Calculate the new width and height
#     new_w = int(h * abs_sin + w * abs_cos)
#     new_h = int(h * abs_cos + w * abs_sin)

#     # Adjust the rotation matrix to account for translation (shifting the image to prevent clipping)
#     matrix[0, 2] += (new_w / 2) - center[0]
#     matrix[1, 2] += (new_h / 2) - center[1]

#     # Rotate the image and resize it to the new size (without clipping)
#     rotated_image = cv2.warpAffine(
#         image,
#         matrix,
#         (new_w, new_h),
#         flags=cv2.INTER_LINEAR,
#         borderMode=cv2.BORDER_REPLICATE,
#     )
#     rotated_contour = cv2.transform(
#         contour.reshape(-1, 1, 2), matrix
#     )  # Transform the contour using the rotation matrix

#     x, y, w, h = cv2.boundingRect(rotated_contour)
#     pad = 3
#     return rotated_image[y + pad : y + h - pad, x + pad : x + w - pad]

# def detect_rois(self):
#     kernel_size = 51
#     background = cv2.GaussianBlur(self.res_matrix, (kernel_size, kernel_size), 0)
#     subtracted = cv2.subtract(self.res_matrix, background)
#     subtracted = cv2.normalize(subtracted, None, 0, 255, cv2.NORM_MINMAX).astype(
#         np.uint8
#     )
#     thresholds = threshold_multiotsu(subtracted, classes=2)
#     mask = np.digitize(subtracted, bins=thresholds).astype(np.uint8)

#     dilated = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)
#     plt.imshow(dilated)
#     plt.show()

# def get_profile(self):
#     pass
#     # nonlinearity_scores = []
#     # for row_index in range(self.res_matrix.shape[0]):
#     #     row_values = self.res_matrix[row_index, :]
#     #     x = np.arange(len(row_values))
#     #     coeffs = np.polyfit(x, row_values, 1)
#     #     fit_line = np.polyval(coeffs, x)
#     #     residuals = row_values - fit_line
#     #     rss = np.sum(residuals**2)
#     #     nonlinearity_scores.append(rss)
#     # filtered_rows = [
#     #     row * ns for row, ns in zip(self.res_matrix, nonlinearity_scores)
#     # ]
#     # signal = np.mean(filtered_rows, axis=0)
