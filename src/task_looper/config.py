import logging

from hazenlib.tasks.acr_slice_thickness import ACRSliceThickness
from hazenlib.tasks.acr_uniformity import ACRUniformity
from hazenlib.tasks.acr_snr import ACRSNR
from hazenlib.tasks.acr_geometric_accuracy import ACRGeometricAccuracy
from hazenlib.tasks.acr_spatial_resolution_rsch_test import ACRSpatialResolution

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

EXPECTED_ORIENTATIONS = ["Ax", "Sag", "Cor"]
EXPECTED_COILS = ["Flex", "HN", "IB"]

IMPLEMENTED_TASKS = (
    "slice_thickness",
    "snr",
    "geometric_accuracy",
    "uniformity",
    "spatial_resolution",
)

TASK_CLASS_REFERENCES = (
    ACRSliceThickness,
    ACRSNR,
    ACRGeometricAccuracy,
    ACRUniformity,
    ACRSpatialResolution,
)

CLASS_AS_STR = (classx.__name__ for classx in TASK_CLASS_REFERENCES)
TASK_TO_CLASS = dict(zip(IMPLEMENTED_TASKS, TASK_CLASS_REFERENCES))
CLASS_STR_TO_TASK = dict(zip(CLASS_AS_STR, IMPLEMENTED_TASKS))
