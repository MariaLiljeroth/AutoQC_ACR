from multiprocessing import Manager
from multiprocessing.managers import BaseProxy

# QUEUE UPDATES ARE CALLED BY PASSING TUPLE (QUEUE ID, QUEUE SUBID, ADDITIONAL ARGS IF NECESSARY...
queue = None


def get_queue() -> BaseProxy:
    """Returns a multiprocessing queue to be used globally throughout application.

    Returns:
        BaseProxy: Centralised multiprocessing queue to handle events throughout frontend and backend.
    """
    global queue
    if queue is None:
        queue = Manager().Queue()
    return queue
