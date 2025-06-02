from backend.smaaf.hazenlib.tasks.acr_slice_thickness import ACRSliceThickness
from backend.smaaf.hazenlib.tasks.acr_snr import ACRSNR
from backend.smaaf.hazenlib.tasks.acr_geometric_accuracy import ACRGeometricAccuracy
from backend.smaaf.hazenlib.tasks.acr_uniformity import ACRUniformity
from backend.smaaf.hazenlib.tasks.acr_spatial_resolution import ACRSpatialResolution
from shared.context import AVAILABLE_TASKS

# Lists of specific string-string and string-class mappings for use throughout backend.
CLASSES = [
    ACRSliceThickness,
    ACRSNR,
    ACRGeometricAccuracy,
    ACRUniformity,
    ACRSpatialResolution,
]
TASK_STR_TO_CLASS = dict(zip(AVAILABLE_TASKS, CLASSES))
CLASS_STR_TO_TASK = dict(zip([c.__name__ for c in CLASSES], AVAILABLE_TASKS))
