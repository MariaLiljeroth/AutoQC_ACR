from typing import Callable

import numpy as np

from scipy.signal import find_peaks, medfilt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

from backend.hazen.hazenlib.utils import XY, Line


class LineSliceThickness(Line):
    """Subclass of Line to implement functionality related to the ACR phantom's ramps

    Instance attributes:
        FWHM (float): Calculated FWHM value for object.
        fitted (XY): Fitted profile from sigmoid fitting model.
    """

    def get_FWHM(self):
        """Calculates the FWHM by fitting a piecewise sigmoid to signal across line"""

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

        # get background pixel values on left and right.

        backgroundL = fitted.y[0]
        backgroundR = fitted.y[-1]

        # calculate FWHM half heights on the left and right.

        halfMaxL = (peak_height - backgroundL) / 2 + backgroundL
        halfMaxR = (peak_height - backgroundR) / 2 + backgroundR

        def simple_interpolate(targetY: float, signal: XY) -> float:
            """Interpolates an XY profile to extract an x-value corresponding
            to a target y-value.

            Args:
                targetY (float): Target y to interpolate x-value from.
                signal (XY): XY signal to interpolate x-value from y-value.

            Returns:
                float: x-value corresponding to targetY.
            """
            crossing_index = np.where(signal.y > targetY)[0][0]
            x1, x2 = signal.x[crossing_index - 1], signal.x[crossing_index]
            y1, y2 = signal.y[crossing_index - 1], signal.y[crossing_index]
            targetX = x1 + (targetY - y1) * (x2 - x1) / (y2 - y1)
            return targetX

        # interpolate x values corresponding to left and right FWHM half-heights.

        xL = simple_interpolate(halfMaxL, fitted)
        xR = simple_interpolate(halfMaxR, fitted[:, ::-1])

        # calculate FWHM and save fitted profile as instance attr.

        self.FWHM = xR - xL
        self.fitted = fitted

    def _fit_piecewise_sigmoid(self) -> XY:
        """Fits a piecewise sigmoid model to the raw signal for
        smoothing purposes.

        Returns:
            XY: Fitted sigmoid model for raw XY signal.
        """

        # get copy of raw signal for smoothing.
        smoothed = self.signal.copy()

        # apply basic smoothing using median and gaussian 1d convolution.
        k = round(len(smoothed.y) / 20)
        if k % 2 == 0:
            k += 1
        smoothed.y = medfilt(smoothed.y, k)
        smoothed.y = gaussian_filter1d(smoothed.y, round(k / 2.5))

        # find global maximum of raw signal.
        peaks, props = find_peaks(
            smoothed.y, height=0, prominence=np.max(smoothed.y).item() / 4
        )
        if len(peaks) == 0:
            self.handle_no_peaks_error()

        heights = props["peak_heights"]
        peak = peaks[np.argmax(heights)]

        def get_specific_sigmoid(wholeData: XY, fitStart: int, fitEnd: int) -> Callable:
            """Fits a specific sigmoid across the whole data x-range.
            Fitting data is specified by args.

            Args:
                wholeData (XY): XY object representing whole signal.
                fitStart (int): index for start of data to use for sigmoid fitting.
                fitEnd (int): index for end of data to use for sigmoid fitting.

            Returns:
                Callable: Functional form of sigmoid for specific fitting range.
            """

            # select data to fit sigmoid from.
            fitData = wholeData[:, fitStart:fitEnd]

            # calculate initial guesses for sigmoid fitting parameters.
            A = np.max(fitData.y) - np.min(fitData.y)
            b = np.min(fitData.y)

            # k estimated from gradient calculations.
            dy = np.diff(fitData.y)
            dx = np.diff(fitData.x)
            dx[dx == 0] = 1e-6
            absDeriv = np.abs(dy / dx)
            absGradMax = np.max(absDeriv)
            k = np.sign(fitData.y[-1] - fitData.y[0]) * absGradMax * 0.5

            x0 = fitData.x[np.argmax(absDeriv)]

            p0 = [A, k, x0, b]

            def sigmoid(
                x: float | np.ndarray, A: float, k: float, x0: float, b: float
            ) -> float:
                """Functional form of sigmoid.

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

            # get fitted sigmoid parameters

            popt, _ = curve_fit(sigmoid, fitData.x, fitData.y, p0=p0, maxfev=10000)

            # return specific fitted sigmoid function for input data.

            def specific_sigmoid(x):
                return sigmoid(x, *popt)

            return specific_sigmoid

        # get sigmoid functions for left and right sides of global maximum by fitting left and right sides of signal.

        sigmoidL_func = get_specific_sigmoid(smoothed, 0, peak)
        sigmoidR_func = get_specific_sigmoid(smoothed, peak, len(smoothed.x))

        # get an XY form of left and right functional sigmoids across whole x-range.

        sigmoidL = XY(smoothed.x, sigmoidL_func(smoothed.x))
        sigmoidR = XY(smoothed.x, sigmoidR_func(smoothed.x))

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

        # get blending function for total x-range

        W = blending_weight(
            smoothed.x, peak, 1 / 20 * peak + 1 / 20 * (len(smoothed.x) - peak)
        )

        # blend XY representations of left and right sigmoid functions together using blending function.

        fitted = XY(smoothed.x, (1 - W) * sigmoidL.y + W * sigmoidR.y)

        return fitted

    def handle_no_peaks_error(self):
        """Handles error where no peaks are detected.
        Signal plot to be generated here (work for future)

        Raises:
            ValueError: No peaks detected error!
        """

        raise ValueError(
            "No peaks found in fitted signal due to poor phantom positioning."
        )
