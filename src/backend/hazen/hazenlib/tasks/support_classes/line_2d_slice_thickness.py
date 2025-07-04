"""
line_2d_slice_thickness.py

This script defines a Line2D subclass, line2DSliceThickness, which implements custom logic related to
the ACR slice thickness task, such as FWHM calculation of the specific profile expected from taking a
pixel value signal across the phantom's slice thickness insert. The effect of noisy data is reduced by
fitting a piecewise sigmoid model to the noisy data.

Written by Nathan Crossley 2025

"""

from typing import Callable

import numpy as np

from scipy.signal import find_peaks, medfilt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

from backend.hazen.hazenlib.tasks.support_classes.line_2d import Line2D
from backend.hazen.hazenlib.tasks.support_classes.series_xy import SeriesXY


class Line2DSliceThickness(Line2D):
    """Subclass of Line2D to implement custom logic related to the
    ACR slice thickness task, such as FWHM calculation from a signal
    of an expected symmetric or asymmetric peak shape.

    Instance attributes:
        FWHM (float): Calculated FWHM value for self.
        fitted (XY): Fitted signal using blended signal model.
    """

    def get_FWHM(self):
        """Calculates the FWHM from a fitted signal, using a piecewise
        sigmoid model to reduce the impact of noisy data."""

        # Raises error if function called without signal attribute existing.
        if not hasattr(self, "signal"):
            raise ValueError("Signal across line has not been computed!")

        # get fitted signal
        fitted = self._fit_piecewise_sigmoid()

        # find peaks of sigmoid model to extract max peak height.
        _, props = find_peaks(
            fitted.y, height=0, prominence=np.ptp(fitted.y).item() / 4
        )
        peak_height = np.max(props["peak_heights"])

        # get background pixel values on left and right sides of peak.
        background_left = fitted.y[0]
        background_right = fitted.y[-1]

        # calculate FWHM half heights on the left and right sides of peak.
        half_max_height_left = (peak_height - background_left) / 2 + background_left
        half_max_height_right = (peak_height - background_right) / 2 + background_right

        def simple_interpolate(y_input: float, signal: SeriesXY) -> float:
            """Interpolates a SeriesXY signal to extract an x-value corresponding
            to a user provided y-value.

            Args:
                y_input (float): y-value to interpolate corresponding x-value from.
                signal (SeriesXY): SeriesXY signal to use for interpolation.

            Returns:
                float: x-value corresponding to input y-value.
            """

            # find index where signal becomes greater than input y-value
            crossing_index = np.where(signal.y > y_input)[0][0]

            # get the x and y-values either side of the crossing index
            x_left, x_right = signal.x[crossing_index - 1], signal.x[crossing_index]
            y_left, y_right = signal.y[crossing_index - 1], signal.y[crossing_index]

            # perform linear interpolation to get x value
            x_output = x_left + (y_input - y_left) * (x_right - x_left) / (
                y_right - y_left
            )

            return x_output

        # interpolate x values corresponding to left and right FWHM half-heights.
        xL = simple_interpolate(half_max_height_left, fitted)
        xR = simple_interpolate(half_max_height_right, fitted[:, ::-1])

        # calculate FWHM and save fitted profile as instance attribute.
        self.FWHM = xR - xL
        self.fitted = fitted

    def _fit_piecewise_sigmoid(self) -> SeriesXY:
        """Fits a piecewise sigmoid model to the raw signal for
        smoothing purposes. This is beneficial over traditional smoothing methods
        and its much more robust to noisy data.

        Returns:
            SeriesXY: SeriesXY object associated with fitted input object.
        """

        # get copy of raw signal for smoothing.
        smoothed = self.signal.copy()

        # apply basic smoothing using median and gaussian 1d convolution.
        k = round(len(smoothed.y) / 20)
        if k % 2 == 0:
            k += 1
        smoothed.y = medfilt(smoothed.y, k)
        smoothed.y = gaussian_filter1d(smoothed.y, round(k / 2.5))

        # find large peaks of signal after basic smoothing.
        peaks, props = find_peaks(
            smoothed.y, height=0, prominence=np.max(smoothed.y).item() / 4
        )

        # throw error if no peaks found (i.e. profile is of unexpected shape)
        if len(peaks) == 0:
            raise ValueError(
                "No peaks found in smoothed slice thickness signal due to poor phantom positioning."
            )

        # get highest peak from signal
        heights = props["peak_heights"]
        peak = peaks[np.argmax(heights)]

        def get_specific_sigmoid(
            raw_signal: SeriesXY, fit_start_idx: int, fit_end_idx: int
        ) -> Callable:
            """Returns a functional sigmoid, with parameters selected by sigmoid fitting
            using raw signal data between specified start and end fitting indices.

            Args:
                raw_signal (SeriesY): SeriesXY object representing whole signal
                    (doesn't all need to be used for fitting).
                fit_start_idx (int): index defining start of data to use for sigmoid fitting.
                fit_end_idx (int): index defining end of data to use for sigmoid fitting.

            Returns:
                Callable: Functional form of sigmoid, tuned using specific fitting range.
            """

            # slice to select data to use for fitting sigmoid
            fitting_data = raw_signal[:, fit_start_idx:fit_end_idx]

            # calculate initial guesses for sigmoid fitting parameters.
            A = np.max(fitting_data.y) - np.min(fitting_data.y)
            b = np.min(fitting_data.y)

            # k estimated from gradient calculations.
            dy = np.diff(fitting_data.y)
            dx = np.diff(fitting_data.x)
            dx[dx == 0] = 1e-6
            abs_deriv = np.abs(dy / dx)
            abs_deriv_max = np.max(abs_deriv)
            k = np.sign(fitting_data.y[-1] - fitting_data.y[0]) * abs_deriv_max * 0.5

            # x_0 estimated from index of maximum abs gradient
            x0 = fitting_data.x[np.argmax(abs_deriv)]

            # store guesses
            p0 = [A, k, x0, b]

            def sigmoid(
                x: float | np.ndarray, A: float, k: float, x0: float, b: float
            ) -> float | np.ndarray:
                """Arbitrary functional form of sigmoid.

                Args:
                    x (np.ndarray): input single value or x-array to get sigmoid y-val(s).
                    A (float): Multiplicative y-scalar param.
                    k (float): Sigmoid steepness param.
                    x0 (float): Sigmoid turning point param.
                    b (float): y-offset param.

                Returns:
                    float | np.ndarray: Returned y-value(s) from sigmoid func.
                """
                exponent = np.clip(-k * (x - x0), -500, 500)
                exp_term = np.exp(exponent)

                return A / (1 + exp_term) + b

            # get fitted sigmoid parameters by using fitting data with initial guesses
            popt, _ = curve_fit(
                sigmoid, fitting_data.x, fitting_data.y, p0=p0, maxfev=10000
            )

            # get callable sigmoid function with calculated best fit parameters.
            def specific_sigmoid(x):
                return sigmoid(x, *popt)

            return specific_sigmoid

        # get sigmoid functions for left and right sides of global maximum by fitting left and right sides of signal.
        sigmoidL_func = get_specific_sigmoid(smoothed, 0, peak)
        sigmoidR_func = get_specific_sigmoid(smoothed, peak, len(smoothed.x))

        # get a SeriesXY object using left and right functional sigmoids across whole x-range.
        sigmoidL = SeriesXY(smoothed.x, sigmoidL_func(smoothed.x))
        sigmoidR = SeriesXY(smoothed.x, sigmoidR_func(smoothed.x))

        def blending_weight(
            x: np.ndarray, transition_x: int, transition_width: int
        ) -> np.ndarray:
            """Generic function for defining a blending function (sigmoid method)

            Args:
                x (np.ndarray): x-array covering whole x-range of functions to blend.
                transition_x (int): x-value around which blending transition should happen.
                transition_width (int): Approximate width of blending transition.

            Returns:
                np.ndarray: blending function for input x-range.
            """
            return 1 / (1 + np.exp(-(x - transition_x) / transition_width))

        # get blending function across total x-range
        W = blending_weight(
            smoothed.x, peak, 1 / 20 * peak + 1 / 20 * (len(smoothed.x) - peak)
        )

        # blend SeriesXY objects for left and right sigmoid functions together using blending function.
        fitted = SeriesXY(smoothed.x, (1 - W) * sigmoidL.y + W * sigmoidR.y)

        return fitted
