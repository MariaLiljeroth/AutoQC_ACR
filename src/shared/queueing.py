"""
queueing.py

This script defines the functions to allow a single multiprocessing queue to be accessed.
Triggers are sent to the queue to communicate between the frontend and the backend,
whilst still allowing the separation of concerns. A specialist QueueTrigger class is provided
for communicating such triggers.

Written by Nathan Crossley 2025.

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


class QueueTrigger:
    """ "Queue representing a particular request, trigger
    or update sent between the frontend and the backend. The
    ID attribute is used to uniquely identify the request and
    the data attribute is used to optionally pass data with the trigger."""

    def __init__(self, ID: str, data: any = None):
        """Initialises QueueTrigger class.

        Args:
            ID (str): Unique identifier for trigger.
            data (any, optional): Additional data to pass with trigger.. Defaults to None.
        """
        self.ID = ID
        self.data = data
