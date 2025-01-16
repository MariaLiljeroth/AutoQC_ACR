import os
import sys
import traceback

sys.path.append(os.path.dirname(os.getcwd()))

import matplotlib.pyplot as plt
import numpy as np
import cv2
import pydicom

from skimage.measure import profile_line
from scipy.signal import find_peaks, peak_widths

import warnings

warnings.filterwarnings("ignore", category=np.ComplexWarning)

from hazenlib.HazenTask import HazenTask
from hazenlib.ACRObject import ACRObject


class Point:
    """Class containing methods and properties of a spatial point, (x, y)"""

    def __init__(self, x, y=None):
        if isinstance(x, (list, tuple)) and len(x) == 2:
            self._xy = np.array(x)
        elif isinstance(x, np.ndarray) and x.shape == (2,):
            self._xy = x
        else:
            raise TypeError("Arguments of Point must be either a list [x, y] or a 2D numpy array.")

    @property
    def xy(self):
        """Getter for xy attribute"""
        return self._xy

    @property
    def x(self):
        """Getter for x value"""
        return self._xy[0]

    @property
    def y(self):
        """Getter for y value"""
        return self._xy[1]

    @property
    def mag(self):
        """Setter for mag attribute"""
        return np.sqrt(self.x**2 + self.y**2)

    def scale(self, f):
        """Scales up xy by factor f"""
        self._xy = self._xy * f

    def copy(self):
        """Returns a copy of the current object"""
        return Point(self._xy)

    def offset(self, targetPoint, sf):
        """Shifts xy towards the target point by vector between them scaled by factor sf"""
        vector = targetPoint - self
        vector.scale(sf)
        offsetPoint = self + vector
        return offsetPoint

    def __add__(self, other):
        if isinstance(other, Point):
            return Point(self._xy + other._xy)
        raise TypeError("Operands must be instances of Point.")

    def __sub__(self, other):
        if isinstance(other, Point):
            return Point(self._xy - other._xy)
        raise TypeError("Operands must be instances of Point.")

    def __str__(self):
        return f"Point({self.x}, {self.y})"


class SignalLine:
    """Class containing the methods and properties of a line placed in physical space"""

    def __init__(self, start, end, referenceImg=None):

        if isinstance(start, Point):
            self._start = start
        else:
            raise TypeError("start must be of type Point")
        if isinstance(end, Point):
            self._end = end
        else:
            raise TypeError("end must be of type Point")

        self._signal = profile_line(
            image=referenceImg,
            src=self._start.xy.astype(int).tolist()[::-1],
            dst=self._end.xy.astype(int).tolist()[::-1],
        )

    @property
    def start(self):
        """Getter for start attribute."""
        return self._start

    @property
    def end(self):
        """Getter for end attribute."""
        return self._end

    @property
    def signal(self):
        """Getter for signal attribute."""
        return self._signal

    @property
    def FWHM_params(self):
        """Getter for FWHM_params attribute."""
        return self._FWHM_params

    @property
    def filteredSignal(self):
        """Getter for filteredSignal attribute."""
        return self._filteredSignal

    def analyse_signal(self, img):
        """Analysis the signal, extracting the FWHM."""
        filteredSignal = self.filter_signal(self._signal, samplingFreq=1000, cutOffFreq=40)
        peakX, peakY = self.extract_peak(signal=filteredSignal, prominence=3, height=0)

        self._FWHM_params = peak_widths(filteredSignal, [peakX], rel_height=0.5)
        self._filteredSignal = filteredSignal

    @staticmethod
    def filter_signal(self, signal: list, samplingFreq: int, cutOffFreq: int) -> list:
        """Filters signal provided in args using fourier analysis approach."""
        samplingFreq = 1000
        cutOffFreq = 40

        fft_signal = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(len(fft_signal), d=1 / samplingFreq)
        fft_signal[np.abs(frequencies) > cutOffFreq] = 0
        filteredSignal = np.fft.ifft(fft_signal)

        return filteredSignal

    @staticmethod
    def extract_peak(signal: list, prominence: int, height: int) -> list[int, float]:
        """Extracts the most prominent peak from the signal in args. Returns its x and y coords."""
        peaks, properties = find_peaks(signal, prominence=prominence, height=height)
        if len(peaks) > 0:
            peakIndex = np.argmax(properties["peak_heights"])
            peakX = peaks[peakIndex]
            peakY = signal[peakX]
        else:
            raise ValueError("No peak detected!")

        return peakX, peakY

    def __str__(self):
        return f"Line(\n\t{self.start},\n\t{self.end}\n)"


class ACRSliceThickness(HazenTask):
    """Class to control slice thickness task."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ACR_obj = ACRObject(self.dcm_list, kwargs)

    def run(self) -> dict:
        """Runs the task"""
        dcm_st = self.ACR_obj.dcm_list[0]

        results = self.init_result_dict()
        results["file"] = self.img_desc(dcm_st)

        try:
            result = self.calc_slice_thickness(dcm_st)
            results["measurement"] = {"slice width mm": round(result, 2)}

        except Exception as e:
            print(
                f"Could not calculate the slice thickness for {self.img_desc(dcm_st)} because of : {e}"
            )
            traceback.print_exc(file=sys.stdout)
            raise Exception(e)

        if self.report:
            results["report_image"] = self.report_files

        return results

    def calc_slice_thickness(self, dcm_st: pydicom.Dataset) -> float:
        """Calculates the slice thickness of the DICOM."""
        if "PixelSpacing" in dcm_st:
            res = dcm_st.PixelSpacing  # In-plane resolution from metadata
        else:
            import hazenlib.utils

            res = hazenlib.utils.GetDicomTag(dcm_st, (0x28, 0x30))
        res = np.mean(res)

        lines = self.place_lines(dcm_st.pixel_array)
        for line in lines:
            line.analyse_signal(dcm_st.pixel_array)

        FWHM1 = float(lines[0].FWHM_params[0])
        FWHM2 = float(lines[1].FWHM_params[0])

        slice_thickness = round(
            0.2 * res * (FWHM1 * FWHM2) / (FWHM1 + FWHM2),
            1,
        )

        if self.report:
            fig, axes = plt.subplots(1, 3, figsize=(16, 8))
            plt.tight_layout()

            axes[0].set_title(
                "Schematic showing line placement within the central insert of the ACR Phantom."
            )
            axes[1].set_title("Signal across blue line.")
            axes[2].set_title("Signal across orange line.")
            axes[0].imshow(dcm_st.pixel_array)

            for i, line in enumerate(lines):
                axes[0].plot(
                    [line.start.x, line.end.x], [line.start.y, line.end.y], lw=2, color=f"C{i}"
                )
                axes[i + 1].plot(line.signal * res, color=f"C{i}", alpha=0.25, label="Raw signal")
                axes[i + 1].plot(line.filteredSignal * res, color=f"C{i}", label="Smoothed signal")
                axes[i + 1].hlines(
                    line.FWHM_params[1] * res,
                    line.FWHM_params[2] * res,
                    line.FWHM_params[3] * res,
                    color=f"C{i}",
                    linestyle="--",
                    label="FWHM",
                )
                axes[i + 1].legend()

            img_path = os.path.realpath(
                os.path.join(self.report_path, f"{self.img_desc(dcm_st)}_slice_thickness.png")
            )
            fig.savefig(img_path, bbox_inches="tight", dpi=600)
            plt.close()
            self.report_files.append(img_path)

        return slice_thickness

    @staticmethod
    def place_lines(img: np.ndarray) -> list[SignalLine]:
        """Places the signal lines within the central insert of the ACR phantom.
        Returns a list of line objects representing the placed lines."""

        # Enhance contrast, otsu threshold and binarize
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(3, 3))
        img_contrastEnhanced = clahe.apply(img)
        _, img_binary = cv2.threshold(
            img_contrastEnhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Find contour of insert - could potentially improve by handling each side individually using Hough Transform?
        contours, _ = cv2.findContours(
            img_binary.astype(np.uint8), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE
        )
        contours = sorted(
            contours,
            key=lambda cont: abs(np.max(cont[:, 0, 0]) - np.min(cont[:, 0, 0])),
            reverse=True,
        )
        insertContour = contours[1]
        insertContour = np.intp(cv2.boxPoints(cv2.minAreaRect(insertContour)))
        insertCorners = [Point(coords) for coords in insertContour]

        # Offset points by 1/3 of distance to nearest point, towards that point
        testPoint = insertCorners[0]
        _, closest, middle, furthest = sorted(
            insertCorners, key=lambda otherPoint: (testPoint - otherPoint).mag
        )
        offset_points = [
            testPoint.copy().offset(closest, sf=1 / 3),
            closest.copy().offset(testPoint, sf=1 / 3),
            middle.copy().offset(furthest, sf=1 / 3),
            furthest.copy().offset(middle, sf=1 / 3),
        ]

        # Offset points by 1/8 of distance to line pair point, towards that point
        testPoint = offset_points[0]
        _, closest, middle, furthest = sorted(
            offset_points, key=lambda otherPoint: (testPoint - otherPoint).mag
        )
        offset_points = [
            testPoint.copy().offset(middle, sf=1 / 30),
            middle.copy().offset(testPoint, sf=1 / 30),
            closest.copy().offset(furthest, sf=1 / 30),
            furthest.copy().offset(closest, sf=1 / 30),
        ]

        # Determine which points to join to form the lines.
        testPoint = offset_points[0]
        _, closest, middle, furthest = sorted(offset_points, key=lambda x: (testPoint - x).mag)
        lines = [
            SignalLine(start=testPoint, end=middle, referenceImg=img),
            SignalLine(start=closest, end=furthest, referenceImg=img),
        ]

        return lines
