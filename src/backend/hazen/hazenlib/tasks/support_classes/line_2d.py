"""
line_2d.py

This script defines a Line2D class, which represents a 2D linear line in Cartesian space,
with a start and end point. Custom line-based logic is also encapsulated such as swapping
the start and end points.

Written by Nathan Crossley 2025

"""

from typing import Self
import numpy as np
from skimage import measure

from backend.hazen.hazenlib.tasks.support_classes.point_2d import Point2D
from backend.hazen.hazenlib.tasks.support_classes.series_xy import SeriesXY


class Line2D:
    """This class represents a 2D linear line in Cartesian space, with a start
    and end point. Custom line-based logic is also encapsulated for the separation
    of concerns."""

    def __init__(self, *args: Point2D):
        """Initialise Line2D object, populating start point, end point and midpoint
        attributes."""

        # Check to see if all positional args are instances of Point2D
        for arg in args:
            if not isinstance(arg, Point2D):
                err = "All positional args should be instances of Point2D"
                raise TypeError(err)

        # Check to see if start and end points are the only two args passed.
        if len(args) != 2:
            err = "Only two positional args should be provided (start and end points)"
            raise ValueError(err)

        # Store start, end and midpoints as instance attributes.
        self.start = args[0]
        self.end = args[1]
        self.midpoint = (self.start + self.end) / 2

    @property
    def length(self) -> float:
        """Property returning the length of the line using Pythagoras' Theorem
        between the start and end points."""

        return np.sqrt(
            (self.start.x - self.end.x) ** 2 + (self.start.y - self.end.y) ** 2
        )

    def get_signal(self, reference_image: np.ndarray):
        """Gets pixel intensity signal across the length of self, using pixel
        values from a provided reference image.

        Args:
            reference_image (np.ndarray): Reference image to obtain pixel values from.

        """

        # measure 1d line profile (signal) between start and end coordinates of self
        signal = measure.profile_line(
            image=reference_image,
            src=self.start[::-1].astype(int).tolist(),
            dst=self.end[::-1].astype(int).tolist(),
        )

        # multiply x by correction factor to ensure that sampled points are a distance of one pixel apart
        x_spaced = range(len(signal)) * self.length / len(signal)

        # define signal attribute, which is an xy-series of spaced x-array and signal values
        self.signal = SeriesXY(x_spaced, signal)

    def get_subline(self, percentage: int | float) -> Self:
        """Returns a "subline" of self. This is a line that shares the
        same unit vector but is reduced in length. Length of subline set
        to be a certain percentage of the length of self, according to value of
        "percentage" arg.

        Args:
            percentage (int | float): controls length of subline. Subline will be "percentage"
                percent of original length. Must be between 0 and 100.


        Returns:
            subline (Self): subline of original line.

        """

        # check that percentage is of numerical type
        if not isinstance(percentage, (int, float)):
            err = f"Percentage should be int of float, not {type(percentage)}."
            raise TypeError(err)

        # check that percentage is within logic bounds i.e. between 0 and 100 %.
        if not 0 < percentage <= 100:
            err = f"Percentage should be in bounds (0, 100] but received {percentage}"
            raise ValueError(err)

        # calculate the percentage that must be taken off each side of the line.
        percentage_off_side = (100 - percentage) / 2

        # work out vector between end and start
        vector = self.start - self.end

        # subtract relevant vector off start and end points to shrink line about centre
        start = self.start - vector * percentage_off_side / 100
        end = self.end + vector * percentage_off_side / 100

        return type(self)(start, end)

    def point_swap(self):
        """Swaps start and end points and reverses associated spatial attributes"""

        # swap start and end points.
        self.start, self.end = self.end, self.start

        # if a signal has been calculated, reverse that too
        if hasattr(self, "signal"):
            self.signal = self.signal[::-1]

    def __iter__(self) -> iter:
        """Get iterable for plotting"""

        points = (self.start, self.end)

        return iter([[p.x for p in points], [p.y for p in points]])

    def __str__(self) -> str:
        """Gets string representation of self for print commands.

        Returns:
            str: String representation of self
        """

        s = f"{self.__class__.__name__}(start={self.start}, end={self.end})"

        return s
