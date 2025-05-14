import multiprocessing as mp

queue = None
"""
QUEUE UPDATES ARE CALLED BY PASSING TUPLE (QUEUE ID, QUEUE SUBID, ADDITIONAL ARGS IF NECESSARY...)
"""


def get_queue():
    global queue
    if queue is None:
        queue = mp.Manager().Queue()
    return queue
