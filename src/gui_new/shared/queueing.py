import multiprocessing as mp
from multiprocessing.managers import BaseProxy

# QUEUE UPDATES ARE CALLED BY PASSING TUPLE (QUEUE ID, QUEUE SUBID, ADDITIONAL ARGS IF NECESSARY...
queue = None


def get_queue() -> BaseProxy:
    """Returns a multiprocessing queue to be used globally throughout application.

    Returns:
        BaseProxy: _description_
    """
    global queue
    if queue is None:
        queue = mp.Manager().Queue()
    return queue
