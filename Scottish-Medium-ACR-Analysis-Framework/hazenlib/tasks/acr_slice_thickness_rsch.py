import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))

import matplotlib.pyplot as plt
import numpy as np
import cv2
import pydicom

from skimage.measure import profile_line
from scipy.signal import find_peaks

import warnings

warnings.filterwarnings("ignore", category=np.ComplexWarning)

from hazenlib.HazenTask import HazenTask
from hazenlib.ACRObject import ACRObject


class Point:
    """Class containing methods and properties of a spatial point, (x, y)"""

    def __init__(self, xy):
        if isinstance(xy, (list, tuple)) and len(xy) == 2:
            self._xy = np.array(xy)
        elif isinstance(xy, np.ndarray) and xy.shape == (2,):
            self._xy = xy
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
        filteredSignal = self.filter_signal(signal=self._signal, samplingFreq=1000, cutOffFreq=40)
        peak, baseL, baseR = self.extract_peak(
            signal=filteredSignal, prominence=1, height=0
        )

        lSlice = filteredSignal[int(baseL.x) : int(peak.x)]
        lBoolMask = lSlice > (peak.y - baseL.y) / 2 + baseL.y
        halfPointL = Point([baseL.x + np.where(lBoolMask)[0][0], lSlice[lBoolMask][0]])

        rSlice = filteredSignal[int(peak.x) : int(baseR.x)]
        rBoolMask = rSlice < (peak.y - baseR.y) / 2 + baseR.y
        halfPointR = Point([peak.x + np.where(rBoolMask)[0][0], rSlice[rBoolMask][0]])

        self._filteredSignal = filteredSignal
        self.peakProps = PeakProperties(peak, halfPointL, halfPointR, baseL, baseR)

    @staticmethod
    def filter_signal(signal: list, samplingFreq: int, cutOffFreq: int) -> list:
        """Filters signal provided in args using fourier analysis approach."""
        samplingFreq = 1000
        cutOffFreq = 40

        fft_signal = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(len(fft_signal), d=1 / samplingFreq)
        fft_signal[np.abs(frequencies) > cutOffFreq] = 0
        filteredSignal = np.fft.ifft(fft_signal).real

        return filteredSignal

    @staticmethod
    def extract_peak(signal: list, prominence: int, height: int) -> list[int, float]:
        """Extracts the most prominent peak from the signal in args. Returns its x and y coords."""
        peaks, properties = find_peaks(signal, prominence=prominence, height=height)
        if len(peaks) > 0:
            peakIndex = np.argmax(properties["peak_heights"])
            peakX = peaks[peakIndex]
            peak = Point([peakX, signal[peakX]])

            lBaseX = properties["left_bases"][peakIndex]
            lBase = Point([lBaseX, signal[lBaseX]])

            rBaseX = properties["right_bases"][peakIndex]
            rBase = Point([rBaseX, signal[rBaseX]])
        else:
            raise ValueError("No peak detected!")

        return peak, lBase, rBase

    def __str__(self):
        return f"Line(\n\t{self.start},\n\t{self.end}\n)"


class PeakProperties:
    def __init__(self, peak, halfPointL, halfPointR, baseL, baseR):
        self.peak = peak
        self.halfPointL = halfPointL
        self.halfPointR = halfPointR
        self.baseL = baseL
        self.baseR = baseR
        self.FWHM = halfPointR.x - halfPointL.x


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

        result = self.calc_slice_thickness(dcm_st)
        results["measurement"] = {"slice width mm": round(result, 2)}

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

        FWHM1, FWHM2 = [float(lines[i].peakProps.FWHM) for i in range(len(lines))]

        slice_thickness = round(
            0.2 * res * (FWHM1 * FWHM2) / (FWHM1 + FWHM2),
            1,
        )

        if self.report:
            fig, axes = plt.subplots(1, 3, figsize=(16, 8))

            axes[0].set_title("Schematic showing line placement within\n the central insert of the ACR Phantom.")
            axes[0].text(x=0.5,
                         y=-0.2, 
                         s=f"Calculated slice thickness: {slice_thickness} mm",
                         transform=axes[0].transAxes, ha="center",
                         fontsize=14,
                         bbox=dict(facecolor="white", boxstyle='round,pad=0.5'))
            axes[0].axis("off")
            axes[1].set_title("Signal across blue line.")
            axes[2].set_title("Signal across orange line.")
            axes[0].imshow(dcm_st.pixel_array)

            for i, line in enumerate(lines):
                axes[0].plot(
                    [line.start.x, line.end.x], [line.start.y, line.end.y], lw=2, color=f"C{i}"
                )
                axes[i + 1].plot(line.signal * res, color=f"C{i}", alpha=0.25, label="Raw signal")
                axes[i + 1].plot(line.filteredSignal * res, color=f"C{i}", label="Smoothed signal")
                axes[i + 1].scatter(
                    line.peakProps.halfPointL.x, line.peakProps.halfPointL.y, color="r"
                )
                axes[i + 1].scatter(line.peakProps.halfPointR.x, line.peakProps.halfPointR.y, color="r")

                graphicsPoints, textPoint = self.gen_FWHM_graphic(line.peakProps)
                axes[i + 1].plot(
                    [point.x for point in graphicsPoints],
                    [point.y for point in graphicsPoints],
                    ls="--",
                    color="r",
                    label="FWHM"
                )
                axes[i + 1].text(
                    textPoint.x, textPoint.y, f"{int(line.peakProps.FWHM)}", ha="center", va="bottom"
                )
                axes[i + 1].legend()

            img_path = os.path.realpath(
                os.path.join(self.report_path, f"{self.img_desc(dcm_st)}_slice_thickness.png")
            )
            plt.tight_layout()
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

    @staticmethod
    def gen_FWHM_graphic(pp: PeakProperties) -> list[Point]:
        halfPointL_low = Point([pp.halfPointL.x, pp.baseL.y])
        halfPointL_high = Point([pp.halfPointL.x, pp.peak.y])

        halfPointR_high = Point([pp.halfPointR.x, pp.peak.y])
        halfPointR_low = Point([pp.halfPointR.x, pp.baseR.y])

        graphicsPoints = [
            pp.baseL,
            halfPointL_low,
            pp.halfPointL,
            halfPointL_high,
            halfPointR_high,
            pp.halfPointR,
            halfPointR_low,
            pp.baseR,
        ]
        textPoint = Point([(pp.halfPointL.x + pp.halfPointR.x) / 2, pp.peak.y + 1])

        return graphicsPoints, textPoint
