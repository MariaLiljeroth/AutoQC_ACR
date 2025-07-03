"""
mappings.py

This script stores any useful dictionary mappings for use across the backend. All mappings take strings as inputs
and return other useful objects or variables.

Written by Nathan Crossley 2025

"""

from backend.hazen.hazenlib.tasks.acr_slice_thickness import ACRSliceThickness
from backend.hazen.hazenlib.tasks.acr_snr import ACRSNR
from backend.hazen.hazenlib.tasks.acr_geometric_accuracy import ACRGeometricAccuracy
from backend.hazen.hazenlib.tasks.acr_uniformity import ACRUniformity
from backend.hazen.hazenlib.tasks.acr_spatial_resolution import ACRSpatialResolution
from shared.context import AVAILABLE_TASKS

# Store classes in a list
CLASSES = [
    ACRSliceThickness,
    ACRSNR,
    ACRGeometricAccuracy,
    ACRUniformity,
    ACRSpatialResolution,
]

# mapping to map a string of a particular Hazen task to the class associated with that task
TASK_STR_TO_CLASS = dict(zip(AVAILABLE_TASKS, CLASSES))

# mapping to map a particular class associated with a Hazen task to its string equivalent
CLASS_STR_TO_TASK = dict(zip([c.__name__ for c in CLASSES], AVAILABLE_TASKS))
