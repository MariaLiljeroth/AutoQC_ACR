"""
context.py

This script provides contextual information for the whole application.
This information is shared across both the frontend and backend.

"""

# The available tasks that AutoQC_ACR supports.

AVAILABLE_TASKS = [
    "Slice Thickness",
    "SNR",
    "Geometric Accuracy",
    "Uniformity",
    "Spatial Resolution",
]

# The expected anatomical planes imaged.
EXPECTED_ORIENTATIONS = ["Ax", "Sag", "Cor"]

# The expected coils tested.
EXPECTED_COILS = ["IB", "HN", "Flex"]
