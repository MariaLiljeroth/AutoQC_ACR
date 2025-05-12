import os
import sys
from typing import Union, Self

sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import matplotlib.pyplot as plt
import numpy as np
import cv2
import pydicom

from skimage.measure import profile_line
from scipy.signal import find_peaks

from hazenlib.HazenTask import HazenTask
from hazenlib.ACRObject import ACRObject
from hazenlib.utils import GetDicomTag, MethodHelper
from hazenlib.Geometry import Line, Point
pass

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
        res = np.mean(
            dcm_st.PixelSpacing if "PixelSpacing" in dcm_st else GetDicomTag(dcm_st, (0x28, 0x30))
        )
        self.lines = self.position_lines(dcm_st)
        for line in self.lines: line.get_signal(refDcm=dcm_st)
        pass

    def position_lines(self, dcm_st: pydicom.Dataset) -> list[Line]:
        """Positions two lines running through the central insert of the slice thickness slice"""
        img_st = dcm_st.pixel_array

        # Get coords of min area rect around ramps insert
        ramps_insert = self.ACR_obj.find_ramps_insert(img_st)
        minAreaRect = cv2.minAreaRect(ramps_insert)
        boxPoints = np.intp(cv2.boxPoints(minAreaRect))

        # Get Line objs for short sides of insert
        corners = [Point(x=row[0], y=row[1]) for row in boxPoints]
        orderedCorners = sorted(corners, key=lambda point: corners[0].get_distance_to(point))
        shortSides = [Line(p1=corners[0], p2=corners[1]), Line(p1=corners[2], p2=corners[3])]

        # Get sublines of short sides and force p1 to be higher in y
        sublines = [line.get_subline(percOfOrig=30) for line in shortSides]
        for line in sublines:
            if line.p1.y < line.p2.y:
                line.point_swap()

        # Define connecting lines
        connectingLines = [
            SignalLine(p1=sublines[0].p1, p2=sublines[1].p1),
            SignalLine(p1=sublines[0].p2, p2=sublines[1].p2)
            ]

        # Final lines are sublines of connecting lines (points converted to int as will be used in image space)
        finalLines = [line.get_subline(percOfOrig=95).points_as_int() for line in connectingLines]

        return finalLines

class SignalLine(Line):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.signal=None

    def get_signal(self, refDcm):
        """Get signal across the line, with reference to a provided dcm"""
        MethodHelper.inspectArgs((refDcm, pydicom.Dataset))
        self._signal = profile_line(
            image=refDcm.pixel_array,
            src=self._p1.xy.tolist()[::-1],
            dst=self._p2.xy.tolist()[::-1],
        )






