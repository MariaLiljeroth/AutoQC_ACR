import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from hazenlib.HazenTask import HazenTask
from hazenlib.ACRObject import ACRObject
from hazenlib.utils import get_dicom_files

import cv2
import matplotlib.pyplot as plt

import numpy as np
from scipy.fft import fft

from scipy import optimize
from scipy.special import sici


class ACRSpatialResolution(HazenTask):
    """Spatial resolution measurement class for DICOM images of the ACR phantom

    Inherits from HazenTask class
    """

    TARGET_THETA_BAR = 3
    SIZE_ROI = 15

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ACR_obj = ACRObject(self.dcm_list, kwargs)

    def run(self):
        self.img_orig = self.ACR_obj.images[0]
        spatial_resolution = self.get_spatial_resolution()

    def get_spatial_resolution(self):
        self.roi, self.roi_centre = self.get_roi()

        self.esf = self.get_esf_raw()
        self.esf_fitted = self.get_esf_fitted()

        self.lsf = self.get_lsf(self.esf)
        self.lsf_fitted = self.get_lsf(self.esf_fitted)

        self.mtf = self.get_mtf(self.lsf)
        self.mtf_fitted = self.get_mtf(self.lsf_fitted)

        plt.imshow(self.roi)
        plt.show()
        plt.scatter(self.esf[0], self.esf[1])
        plt.show()

    def get_roi(self, size_ROI: int = 15):
        contour_bar = self.get_contour_bar(self.img_orig)
        img_rotated = self.rotate_rel_to_bar(contour_bar)
        contour_bar_rotated = self.get_contour_bar(img_rotated)
        roi_centre = self.define_ROI_centre(contour_bar_rotated)

        x1 = int(roi_centre[0] - size_ROI // 2)
        x2 = int(roi_centre[0] + size_ROI // 2)
        y1 = int(roi_centre[1] - size_ROI // 2)
        y2 = int(roi_centre[1] + size_ROI // 2)

        roi = img_rotated[y1:y2, x1:x2]

        return roi, roi_centre

    @staticmethod
    def get_contour_bar(img):
        normalised = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        contrast_enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3)).apply(
            normalised
        )
        _, thresholded = cv2.threshold(
            contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        contours, _ = cv2.findContours(
            thresholded.astype(np.uint8),
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_NONE,
        )

        def get_aspect_ratio(contour):
            _, (width, height), _ = cv2.minAreaRect(contour)
            return min(width, height) / max(width, height)

        # filter out tiny contours from noise
        threshold_area = 15 * 15
        filtered_contours = [
            c for c in contours if cv2.contourArea(c) >= threshold_area
        ]
        # select central insert
        contour_bar = sorted(
            filtered_contours,
            key=lambda c: get_aspect_ratio(c),
        )[0]
        return contour_bar

    def rotate_rel_to_bar(self, contour_bar) -> np.ndarray:
        _, (w, h), theta = cv2.minAreaRect(contour_bar)
        if w < h:
            theta = theta - 90

        theta_to_apply = theta - self.TARGET_THETA_BAR

        (img_h, img_w) = self.img_orig.shape[:2]
        # Calculate the center of the image
        center = (img_w // 2, img_h // 2)

        # Calculate the rotation matrix
        matrix = cv2.getRotationMatrix2D(center, theta_to_apply, 1.0)

        # Get the new bounding box dimensions after rotation
        abs_cos = abs(matrix[0, 0])
        abs_sin = abs(matrix[0, 1])

        # Calculate the new width and height
        new_w = int(img_h * abs_sin + img_w * abs_cos)
        new_h = int(img_h * abs_cos + img_w * abs_sin)

        # Adjust the rotation matrix to account for translation (shifting the image to prevent clipping)
        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]

        # Rotate the image and resize it to the new size (without clipping)
        img_rotated = cv2.warpAffine(
            self.img_orig,
            matrix,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return img_rotated

    @staticmethod
    def define_ROI_centre(contour_bar_rotated):
        corners = np.intp(cv2.boxPoints(cv2.minAreaRect(contour_bar_rotated)))
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
        theta_rad = np.deg2rad(self.TARGET_THETA_BAR)

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

        popt, _ = optimize.curve_fit(
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
        mtf = np.abs(fft(lsf[1]))
        mtf /= np.max(mtf)
        return mtf


from tkinter import filedialog

path = filedialog.askdirectory()
obj = ACRSpatialResolution(input_data=get_dicom_files(path))
obj.run()


# @classmethod
# def get_res_matrix(cls, img):
#     contour = cls.locate_insert(img)
#     res_matrix = cls.crop_within_contour_axis_aligned(img, contour)
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
