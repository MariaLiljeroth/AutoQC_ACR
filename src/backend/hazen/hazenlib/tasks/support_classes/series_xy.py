"""
series_xy.py

This script defines an np.ndarray subclass, SeriesXY, which acts as a convenience
class for plotting 2D numpy arrays. x and y arrays can be accessed conveniently using
x and y attributes.

Written by Nathan Crossley 2025

"""

from typing import Self
import numpy as np


class SeriesXY(np.ndarray):
    """Subclass of np.ndarray used to make plotting of 2D arrays more
    convenient. This is useful to make more complicated code more readable
    in Hazen task code."""

    def __new__(cls, *args: list[int | float]):
        """Instantiate SeriesXY class, returning a numpy array viewed
        as an object of this class."""

        # check that input x and y arrays have the same length
        if len(set(map(len, args))) != 1:
            raise ValueError("All input arrays must have the same length.")

        # create a 2d array to store state
        arr = np.array(args)

        # check that x and y arrays are both 1D
        if arr.ndim != 2 or arr.shape[0] != 2:
            raise ValueError("Input args should be two 1D arrays.")

        return np.asarray(arr).view(cls)

    @property
    def x(self) -> Self:
        """Property for x array of plotting series"""
        return self[0]

    @property
    def y(self) -> Self:
        """Property for x array of plotting series"""
        return self[1]

    @y.setter
    def y(self, val: np.ndarray):
        """Setter for y property"""

        # check that array that is to be set is the same length as existing array
        if isinstance(val, (np.ndarray, list)):
            if len(val) != len(self.y):
                raise ValueError("Cannot modify y due to invalid shape of new y value.")
        else:
            raise TypeError(
                "Cannot modify y using an object not of type list or np.ndarray."
            )
        self[1] = val
