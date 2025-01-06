import pydicom
from tkinter import filedialog, Tk
import matplotlib.pyplot as plt
import numpy as np
import cv2

from skimage.measure import profile_line
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

class Point:
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


class Line:
    def __init__(self, start, end, referenceImg=None):
        # Initialise coordinate attributes
        if isinstance(start, Point):
            self._start = start
        else:
            raise TypeError("start must be of type Point")
        if isinstance(end, Point):
            self._end = end
        else:
            raise TypeError("end must be of type Point")
        self._length = np.sqrt((self.start.x - self.end.x) ** 2 + (self.start.y - self.end.y) ** 2)

        # Initialise profile
        profile = profile_line(
            image=referenceImg,
            src=self._start.xy.astype(int).tolist()[::-1],
            dst=self._end.xy.astype(int).tolist()[::-1],
        )
        self._profile = gaussian_filter1d(profile, sigma=len(profile) / 50)

    @property
    def start(self):
        """Getter for start attribute"""
        return self._start

    @property
    def end(self):
        """Getter for end attribute"""
        return self._end

    @property
    def length(self):
        """Getter for length attribute"""
        return self._length

    @property
    def profile(self):
        """Getter for profile attribute"""
        return self._profile

    @property
    def binarizedProfile(self):
        """Getter for binarized profile attribute"""
        return self._binarizedProfile

    @property
    def FWHM(self):
        """Getter for FWHM attribute"""
        return self._FWHM

    def analyse_profile(self, img):
        """Calculate profile based on start and end points using image provided in args"""

        backgroundY = self.determine_background(self._profile)
        peakX, peakY = self.extract_peaks(self._profile)
        lim = (peakY - np.min(self._profile)) / 2 + np.min(self._profile)

        binarizedProfile = np.where(self._profile >= lim, peakY, backgroundY)
        FWHM = binarizedProfile.tolist().count(peakY)

        self._binarizedProfile = binarizedProfile
        # Correct for distance to pixel ratio.
        self._FWHM = FWHM * self.length / len(self._profile)

    @staticmethod
    def determine_background(profile):
        approxBackground = profile[profile < np.min(profile) + 0.05 * abs(np.max(profile) - np.min(profile))]
        background = np.mean(approxBackground)
        return background

    @staticmethod
    def extract_peaks(profile):
        peaks, properties = find_peaks(profile, prominence=3, height=0)
        if len(peaks) == 1:
            peakX = peaks[0]
            peakY = properties["peak_heights"][0]
        elif len(peaks) > 1:
            peakX = np.mean(peaks)
            peakY = np.mean(properties["peak_heights"])
        else:
            raise ValueError("No peaks detected!")
        return peakX, peakY

    def plot_line(self):
        """Plots the line"""
        plt.plot([self.start.x, self.end.x], [self.start.y, self.end.y], lw=1)

    def plot_profiles(self):
        plt.plot(self._profile)
        plt.plot(self._binarizedProfile, ls="--")

    def __str__(self):
        return f"Line(\n\t{self.start},\n\t{self.end}\n)"


class ACRSliceThickness:
    def run(self):
        self.img = self.extract_image()
        self.lines = self.find_lines()

        for line in self.lines:
            line.analyse_profile(self.img)
            line.plot_profiles()
            plt.show()

        for line in self.lines:
            line.plot_line()
        plt.imshow(self.img)
        plt.show()
        
        self.report_results()

    @staticmethod
    def extract_image():
        root = Tk()
        root.withdraw()
        root.call('wm', 'attributes', '.', '-topmost', True)
        dcmPath = filedialog.askopenfilename(title="Please select a DICOM file.", parent = root)
        root.destroy()
        ds = pydicom.dcmread(dcmPath)
        img = ds.pixel_array

        return img

    def find_lines(self):
        """Finds the start and end coordinates of the profile lines

        Args:
            img (np.ndarray): Pixel array from DICOM image.

        Returns:
            lines (list of Line): list of line objects representing the placed lines on the image.
        """
        # Applying canny edge to uint8 representation of imgage and dilating.
        img_uint8 = np.uint8(cv2.normalize(self.img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
        img_uint8 = cv2.GaussianBlur(img_uint8, ksize=(15, 15), sigmaX=0, sigmaY=0)
        canny = cv2.dilate(
            cv2.Canny(img_uint8, threshold1=25, threshold2=50), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        )

        # Find Contours. Sort by horizontal span and select second in list (which will be central insert)
        contours, _ = cv2.findContours(canny, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda cont: abs(np.max(cont[:, 0, 0]) - np.min(cont[:, 0, 0])), reverse=True)
        rectCont = np.intp(cv2.boxPoints(cv2.minAreaRect(contours[1])))
        rectPoints = [Point(coords) for coords in rectCont]

        # Offset points by 1/3 of distance to nearest point, towards that point
        testPoint = rectPoints[0]
        _, closest, middle, furthest = sorted(rectPoints, key=lambda otherPoint: (testPoint - otherPoint).mag)
        offset_points = [
            testPoint.copy().offset(closest, sf=1 / 3),
            closest.copy().offset(testPoint, sf=1 / 3),
            middle.copy().offset(furthest, sf=1 / 3),
            furthest.copy().offset(middle, sf=1 / 3),
        ]

        # Offset points by 1/8 of distance to line pair point, towards that point
        testPoint = offset_points[0]
        _, closest, middle, furthest = sorted(offset_points, key=lambda otherPoint: (testPoint - otherPoint).mag)
        offset_points = [
            testPoint.copy().offset(middle, sf=1 / 50),
            middle.copy().offset(testPoint, sf=1 / 50),
            closest.copy().offset(furthest, sf=1 / 50),
            furthest.copy().offset(closest, sf=1 / 50),
        ]

        # Determine which points to join to form the lines.
        testPoint = offset_points[0]
        _, closest, middle, furthest = sorted(offset_points, key=lambda x: (testPoint - x).mag)
        lines = [Line(start=testPoint, end=middle, referenceImg=self.img), Line(start=closest, end=furthest, referenceImg=self.img)]

        return lines

    def report_results(self):
        for line in self.lines:
            print(f"FWHM: {line.FWHM:.1f} pixels")


task = ACRSliceThickness()
task.run()
