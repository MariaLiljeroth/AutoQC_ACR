"""
queueing.py

This script defines the functions to allow a single multiprocessing queue to be accessed.
Progress updates are sent to the queue to communicate between the frontend and the backend,
whilst still allowing the separation of concerns.

Queue updates should be passed as tuples with (GENERIC_ID, SPECIFIC_ID, ADDITIONAL_ARGS...)
    where GENERIC_ID is a general identifier for the nature of the update (e.g. TASK_COMPLETE to indicate that a task has been completed).
    and SPECIFIC_ID is a specific identifier for the nature of the update (e.g. DICOM_SORTING to indicate that the DICOM sorting process has been completed).
    and ADDITIONAL_ARGS is a placeholder for any additional args or data that needs to be passed through the queue.

Written by Nathan Crossley 2025

"""

from multiprocessing import Manager
from multiprocessing.managers import BaseProxy

# queue is initially set to None at the module level to ensure that only one queue is created
# throughout the whole AutoQC_ACR running process.
queue = None


def get_queue() -> BaseProxy:
    """Returns a multiprocessing queue to be used globally throughout application.
    The queue can be used to communicate signals and data between the frontend and the backend.

    Returns:
        BaseProxy: Centralised multiprocessing queue to handle events throughout frontend and backend.
    """
    global queue
    if queue is None:
        queue = Manager().Queue()
    return queue
