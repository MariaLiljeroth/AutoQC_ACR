import os
import cv2 as cv
import pydicom
import imutils
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pathlib

from collections import defaultdict
from skimage import filters, measure
from typing import Union, TypeVar

import hazenlib.exceptions as exc
from hazenlib.logger import logger

P = TypeVar("P", bound="Point")
L = TypeVar("L", bound="Line")
xy = TypeVar("xy", bound="XY")

def GetDicomTag(dcm, tag):
    for elem in dcm.iterall():
        if elem.tag == tag:
            return elem.value


def get_dicom_files(folder: str, sort=False) -> list:
    if sort:
        file_list = [
            os.path.join(folder, x)
            for x in os.listdir(folder)
            if is_dicom_file(os.path.join(folder, x))
        ]
        file_list.sort(key=lambda x: pydicom.dcmread(x).InstanceNumber)
    else:
        file_list = [
            os.path.join(folder, x)
            for x in os.listdir(folder)
            if is_dicom_file(os.path.join(folder, x))
        ]
    return file_list


def is_dicom_file(filename):
    """
    Util function to check if file is a dicom file
    the first 128 bytes are preamble
    the next 4 bytes should contain DICM otherwise it is not a dicom

    :param filename: file to check for the DICM header block
    :type filename: str
    :returns: True if it is a dicom file
    """
    if os.path.isfile(filename) == False:
        return False

    file_stream = open(filename, "rb")
    file_stream.seek(128)
    data = file_stream.read(4)
    file_stream.close()
    if data == b"DICM":
        return True
    else:
        return False


def is_enhanced_dicom(dcm: pydicom.Dataset) -> bool:
    """

    Parameters
    ----------
    dcm

    Returns
    -------
    bool

    Raises
    ------
    Exception
     Unrecognised SOPClassUID

    """

    if dcm.SOPClassUID == "1.2.840.10008.5.1.4.1.1.4.1":
        return True
    elif dcm.SOPClassUID == "1.2.840.10008.5.1.4.1.1.4":
        return False
    else:
        raise Exception("Unrecognised SOPClassUID")


def get_manufacturer(dcm: pydicom.Dataset) -> str:
    supported = ["ge", "siemens", "philips", "toshiba", "canon"]
    manufacturer = dcm.Manufacturer.lower()
    for item in supported:
        if item in manufacturer:
            return item

    raise Exception(f"{manufacturer} not recognised manufacturer")


def get_average(dcm: pydicom.Dataset) -> float:
    try:
        if is_enhanced_dicom(dcm):
            averages = dcm.SharedFunctionalGroupsSequence[0].MRAveragesSequence[0].NumberOfAverages
        else:
            averages = dcm.NumberOfAverages
    except:
        averages = GetDicomTag(dcm, (0x18, 0x83))

    return averages


def get_bandwidth(dcm: pydicom.Dataset) -> float:
    """
    Returns PixelBandwidth

    Parameters
    ----------
    dcm: pydicom.Dataset

    Returns
    -------
    bandwidth: float
    """
    if "PixelBandwidth" in dcm:
        bandwidth = dcm.PixelBandwidth
    else:
        bandwidth = GetDicomTag(dcm, (0x18, 0x95))
    return bandwidth


def get_num_of_frames(dcm: pydicom.Dataset) -> int:
    """
    Returns number of frames of dicom object

    Parameters
    ----------
    dcm: pydicom.Dataset
        DICOM object

    Returns
    -------

    """
    if len(dcm.pixel_array.shape) > 2:
        return dcm.pixel_array.shape[0]
    elif len(dcm.pixel_array.shape) == 2:
        return 1


def get_slice_thickness(dcm: pydicom.Dataset) -> float:
    if is_enhanced_dicom(dcm):
        try:
            slice_thickness = (
                dcm.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].SliceThickness
            )
        except AttributeError:
            slice_thickness = (
                dcm.PerFrameFunctionalGroupsSequence[0].Private_2005_140f[0].SliceThickness
            )
        except Exception:
            raise Exception("Unrecognised metadata Field for Slice Thickness")
    else:
        slice_thickness = dcm.SliceThickness

    return slice_thickness


def get_pixel_size(dcm: pydicom.Dataset) -> tuple[float, float]:
    manufacturer = get_manufacturer(dcm)
    try:
        if is_enhanced_dicom(dcm):
            dx, dy = dcm.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing
        else:
            dx, dy = dcm.PixelSpacing
    except:
        print("Warning: Could not find PixelSpacing..")
        if "ge" in manufacturer:
            fov = get_field_of_view(dcm)
            dx = fov / dcm.Columns
            dy = fov / dcm.Rows
        else:
            raise Exception("Manufacturer not recognised")

    return dx, dy


def get_TR(dcm: pydicom.Dataset) -> float:
    """
    Returns Repetition Time (TR)

    Parameters
    ----------
    dcm: pydicom.Dataset

    Returns
    -------
    TR: float
    """

    try:
        TR = dcm.RepetitionTime
    except:
        print("Warning: Could not find Repetition Time. Using default value of 1000 ms")
        TR = 1000
    return TR


def get_rows(dcm: pydicom.Dataset) -> float:
    """
    Returns number of image rows (rows)

    Parameters
    ----------
    dcm: pydicom.Dataset

    Returns
    -------
    rows: float
    """
    try:
        rows = dcm.Rows
    except:
        print("Warning: Could not find Number of matrix rows. Using default value of 256")
        rows = 256

    return rows


def get_columns(dcm: pydicom.Dataset) -> float:
    """
    Returns number of image columns (columns)

    Parameters
    ----------
    dcm: pydicom.Dataset

    Returns
    -------
    columns: float
    """
    try:
        columns = dcm.Columns
    except:
        print("Warning: Could not find matrix size (columns). Using default value of 256.")
        columns = 256
    return columns


def get_field_of_view(dcm: pydicom.Dataset):
    # assumes square pixels
    manufacturer = get_manufacturer(dcm)

    if "ge" in manufacturer:
        fov = dcm[0x19, 0x101E].value
    elif "siemens" in manufacturer:
        fov = dcm.Columns * dcm.PixelSpacing[0]
    elif "philips" in manufacturer:
        if is_enhanced_dicom(dcm):
            fov = (
                dcm.Columns
                * dcm.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing[0]
            )
        else:
            fov = dcm.Columns * dcm.PixelSpacing[0]
    elif "toshiba" in manufacturer:
        fov = dcm.Columns * dcm.PixelSpacing[0]
    else:
        raise NotImplementedError(
            "Manufacturer not ge,siemens, toshiba or philips so FOV cannot be calculated."
        )

    return fov


def get_image_orientation(iop):
    """
    From http://dicomiseasy.blogspot.com/2013/06/getting-oriented-using-image-plane.html
    Args:
        iop:

    Returns:

    """
    iop_round = [round(x) for x in iop]
    plane = np.cross(iop_round[0:3], iop_round[3:6])
    #Orient=elem=iop[0x0020,0x0037]
    #plane = np.cross(Orient[0:3], Orient[3:6])
    plane = [abs(x) for x in plane]
    if plane[0] == 1:
        return "Sagittal"
    elif plane[1] == 1:
        return "Coronal"
    elif plane[2] == 1:
        return "Transverse"


def rescale_to_byte(array):
    """
    WARNING: This function normalises/equalises the histogram. This might have unintended consequences.
    Args:
        array:

    Returns:

    """
    image_histogram, bins = np.histogram(array.flatten(), 255)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(array.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(array.shape).astype("uint8")


class Rod:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Rod: {self.x}, {self.y}"

    def __str__(self):
        return f"Rod: {self.x}, {self.y}"

    @property
    def centroid(self):
        return self.x, self.y

    def __lt__(self, other):
        """Using "reading order" in a coordinate system where 0,0 is bottom left"""
        try:
            x0, y0 = self.centroid
            x1, y1 = other.centroid
            return (-y0, x0) < (-y1, x1)
        except AttributeError:
            return NotImplemented

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class ShapeDetector:
    """
    This class is largely adapted from https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/

    """

    def __init__(self, arr):
        self.arr = arr
        self.contours = None
        self.shapes = defaultdict(list)
        self.blurred = None
        self.thresh = None

    def find_contours(self):
        # convert the resized image to grayscale, blur it slightly, and threshold it
        self.blurred = cv.GaussianBlur(self.arr.copy(), (5, 5), 0)  # magic numbers

        optimal_threshold = filters.threshold_li(
            self.blurred, initial_guess=np.quantile(self.blurred, 0.50)
        )
        self.thresh = np.where(self.blurred > optimal_threshold, 255, 0).astype(np.uint8)

        # have to convert type for find contours
        contours = cv.findContours(self.thresh, cv.RETR_TREE, 1)
        self.contours = imutils.grab_contours(contours)
        # rep = cv.drawContours(self.arr.copy(), [self.contours[0]], -1, color=(0, 255, 0), thickness=5)
        # plt.imshow(rep)
        # plt.title("rep")
        # plt.colorbar()
        # plt.show()

    def detect(self):
        for c in self.contours:
            # initialize the shape name and approximate the contour
            peri = cv.arcLength(c, True)
            if peri < 100:
                # ignore small shapes, magic number is complete guess
                continue
            approx = cv.approxPolyDP(c, 0.04 * peri, True)

            # if the shape is a triangle, it will have 3 vertices
            if len(approx) == 3:
                shape = "triangle"

            # if the shape has 4 vertices, it is either a square or
            # a rectangle
            elif len(approx) == 4:
                shape = "rectangle"

            # if the shape is a pentagon, it will have 5 vertices
            elif len(approx) == 5:
                shape = "pentagon"

            # otherwise, we assume the shape is a circle
            else:
                shape = "circle"

            # return the name of the shape
            self.shapes[shape].append(c)

    def get_shape(self, shape):
        self.find_contours()
        self.detect()

        if shape not in self.shapes.keys():
            # print(self.shapes.keys())
            raise exc.ShapeDetectionError(shape)

        if len(self.shapes[shape]) > 1:
            shapes = [{shape: len(contours)} for shape, contours in self.shapes.items()]
            raise exc.MultipleShapesError(shapes)

        contour = self.shapes[shape][0]
        if shape == "circle":
            # (x,y) is centre of circle, in x, y coordinates. x=column, y=row.
            (x, y), r = cv.minEnclosingCircle(contour)
            return x, y, r

        # Outputs in below code chosen to match cv.minAreaRect output
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#b-rotated-rectangle
        # (x,y) is top-left of rectangle, in x, y coordinates. x=column, y=row.

        if shape == "rectangle" or shape == "square":
            (x, y), size, angle = cv.minAreaRect(contour)
            # OpenCV v4.5 adjustment
            # - cv.minAreaRect() output tuple order changed since v3.4
            # - swap size order & rotate angle by -90
            size = (size[1], size[0])
            angle = angle - 90
            return (x, y), size, angle


class XY(np.ndarray):
    """Class for 2D numpy array for plotting"""

    def __new__(cls, *args: list[Union[int, float]]):
        """Initialise class"""
        if len(set(map(len, args))) != 1:
            raise ValueError("All input arrays must have the same length")
        arr = np.array(args)
        if arr.ndim != 2 or arr.shape[0] != 2:
            raise ValueError("args of XY should be two 1d arrays")
        return np.asarray(arr).view(cls)

    @property
    def x(self) -> xy:
        """Property for x array of plotting series"""
        return self[0]

    @property
    def y(self) -> xy:
        """Property for x array of plotting series"""
        return self[1]

    @y.setter
    def y(self, val: np.ndarray):
        """Setter for y property"""
        if isinstance(val, (np.ndarray, list)):
            if len(val) != len(self.y):
                raise ValueError("Cannot modify shape of XY.y")
        else:
            raise TypeError("Expected input to be either a list or numpy.ndarray")
        self[1] = val


class Point(np.ndarray):
    """Class for 2D spatial point"""

    def __new__(cls, *args: Union[int, float]) -> np.ndarray:
        """Initialise the point class."""
        supported_dims = {2}
        if len(args) not in supported_dims:
            err = (
                f"Only {' or '.join(str(d) + 'D' for d in supported_dims)}"
                " points are supported"
            )
            raise NotImplementedError(err)

        return np.asarray(args, dtype=float).view(cls)

    @property
    def x(self) -> float:
        """Property for x coordinate"""
        return self[0]

    @property
    def y(self) -> float:
        """property for y coordinate"""
        return self[1]

    def get_distance_to(self, other: P) -> float:
        """Calculates distance between two point objects

        Args:
            other: Point to calculate distance to

        Returns:
            dist: The distance between the points

        """

        if not isinstance(other, Point):
            err = f"other should be Point and not {type(other)}"
            raise TypeError(err)
        return np.sqrt(np.sum((self - other)**2)).item()

    def __iter__(self):
        """Get iterable for plotting"""
        return iter([self.x, self.y])


class Line:
    """Class for line joining two points"""

    def __init__(self, *args: Point) -> None:
        """Initialise Line object"""

        for arg in args:
            if not isinstance(arg, Point):
                err = "All positional args should be instances of Point"
                raise TypeError(err)

        if len(args) != 2:
            err = "Only two positional args should be provided (start and end points)"
            raise ValueError(err)

        self.start = args[0]
        self.end = args[1]
        self.midpoint = (self.start + self.end) / 2

    @property
    def length(self):
        """Property for length of line"""
        return np.sqrt((self.start.x - self.end.x)**2 + (self.start.y - self.end.y)**2)

    def get_signal(self, refImg: np.ndarray) -> None:
        """Gets signal across line using pixel values from reference image

        Args:
            refImg (np.ndarray): Reference image for obtaining pixel values

        """
        signal = measure.profile_line(
            image=refImg,
            src=self.start[::-1].astype(int).tolist(),
            dst=self.end[::-1].astype(int).tolist(),
        )

        # multiply x by correction factor to ensure that sampled points are a distance of one pixel apart
        self.signal = XY(range(len(signal)) * self.length/len(signal), signal)

    def get_subline(self, perc: Union[int, float]) -> L:
        """Returns a "subline" of self.
        This is a line that shares the same unit vector but is reduced in length.
        Length of subline set to be "perc" percent of the length of self.

        Args:
            perc (int, float): controls length of subline. Subline will be "perc" percent of original length.

        Returns:
            subline (Line): subline of original line

        """
        if not isinstance(perc, (int, float)):
            err = f"perc should be int of float, not {type(perc)}."
            raise TypeError(err)

        if not 0 < perc <= 100:
            err = f"perc should be in bounds (0, 100] but received {perc}"
            raise ValueError(err)

        percOffSide = (100 - perc) / 2
        vector = self.start - self.end

        start = self.start - vector * percOffSide / 100
        end = self.end + vector * percOffSide / 100

        return type(self)(start, end)

    def point_swap(self):
        """Swaps start and end points and reverses associated attributes
        Args:
            None
        Returns:
            None
        """
        self.start, self.end = self.end, self.start
        if hasattr(self, "signal"):
            self.signal = self.signal[::-1]


    def __iter__(self) -> iter:
        """Get iterable for plotting"""
        points = (self.start, self.end)
        return iter([[p.x for p in points], [p.y for p in points]])

    def __str__(self):
        """Get string representation"""
        s = f"Line(start={self.start}, end={self.end})"
        return s


