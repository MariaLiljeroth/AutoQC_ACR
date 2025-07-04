"""
point_2d.py

This script defines an np.ndarray subclass, Point2D, which represents a 2D point in Cartesian space.
This is useful to improve readability of more complex code within Hazen tasks.

Written by Nathan Crossley 2025

With advise from Alex Drysdale
"""

from typing import Self
import numpy as np


class Point2D(np.ndarray):
    """Subclass of np.ndarray to represent a 2D point in Cartesian space.
    This is useful to improve readability of more complex code within Hazen tasks.
    """

    def __new__(cls, *args: int | float) -> Self:
        """Instantiates the Point2D class, returning a numpy array viewed
        as an object of this class.

        Returns:
            Self: Instantiated Point2D object.
        """

        # define supported dimensions for class
        supported_dims = {2}

        # raise NotImplementedError if dimensions not supported
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

    def get_distance_to(self, other: Self) -> float:
        """Calculates the linear distance between two Point2D objects
        using Pythagoras' Theorem.

        Args:
            other: Other Point2D object to calculate the distance to.

        Returns:
            dist: The linear distance between the two points.

        """

        # ensure that the other point passed is of type Point2D
        if not isinstance(other, Point2D):
            err = f"other should be Point and not {type(other)}"
            raise TypeError(err)

        # calculate hypotenuse by Pythagoras
        dist = np.sqrt(np.sum((self - other) ** 2)).item()

        return dist

    def __iter__(self):
        """Get iterable for plotting"""
        return iter([self.x, self.y])
