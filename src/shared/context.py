"""
context.py

This script provides contextual information for the whole application.
This information is shared across both the frontend and backend.

Written by Nathan Crossley, 2025.

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

# The manufacturers available for testing.
IMPLEMENTED_MANUFACTURERS = ["Siemens", "Philips", "GE", "Canon"]
