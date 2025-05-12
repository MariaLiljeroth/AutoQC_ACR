import multiprocessing as mp

queue = None


def get_queue():
    global queue
    if queue is None:
        queue = mp.Manager().Queue()
    return queue
