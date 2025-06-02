from pathlib import Path

import multiprocessing as mp
from multiprocessing.managers import BaseProxy

from backend.mappings import TASK_STR_TO_CLASS, CLASS_STR_TO_TASK
from backend.smaaf.hazenlib.utils import get_dicom_files
from backend.utils import nested_dict, defaultdict_to_dict, substring_matcher
from shared.queueing import get_queue
from shared.context import EXPECTED_ORIENTATIONS, EXPECTED_COILS


def run_tasks(task_args: list[tuple[Path, Path, str]]):
    """Runs relevant task for each set of args in task_args.

    Args:
        task_args (list[tuple[Path, Path, str]]): List of args to run each task with.
            Each tuple contains the input and output directories and the task to run.
    """
    # Get queue and add to each task_args tuple as cannot directly access a queue within a process.
    queue = get_queue()
    task_args_with_queue = [
        (in_subdir, out_subdir, task, queue, 1 / len(task_args) * 100)
        for in_subdir, out_subdir, task in task_args
    ]

    # Create a pool of processes to run tasks in parallel.
    cpu_cores = mp.cpu_count()
    pool = mp.Pool(
        processes=cpu_cores // 2 if len(task_args_with_queue) > cpu_cores // 2 else 1
    )
    try:
        results = pool.starmap(run_solo_task_on_folder, task_args_with_queue)
    finally:
        pool.close()
        pool.join()

    # Format results in nested dictionary structure and send via queue.
    formatted_results = format_results(results)
    get_queue().put(("TASK_COMPLETE", "TASK_RUNNING", formatted_results))


def run_solo_task_on_folder(
    in_subdir: Path, out_subdir: Path, task: str, queue: BaseProxy, perc: float
) -> dict:
    """Returns a single task for a specific arg set in task_args.
    Returns the raw Hazen output for the task (which is in dict structure).

    Args:
        in_subdir (Path): Specific input subdirectory for task.
        out_subdir (Path): Specific output subdirectory for task.
        task (str): Task to run.
        queue (BaseProxy): Queue to send progress updates to.
        perc (float): Float value to add to current state of progress bar.

    Returns:
        dict: Raw Hazen result for specific task_args tuple.
    """
    snr_helper = in_subdir / "helper_data_set"
    kwargs = {
        "input_data": get_dicom_files(in_subdir.resolve()),
        "report_dir": out_subdir.resolve(),
        "report": True,
        "MediumACRPhantom": True,
    }

    if task == "SNR" and snr_helper.exists():
        kwargs["subtract"] = snr_helper

    task_obj = TASK_STR_TO_CLASS[task](**kwargs)
    result = task_obj.run()
    queue.put(("PROGRESS_BAR_UPDATE", "TASK_RUNNING", perc))
    return result


def format_results(results: dict) -> dict:
    """Formats the results from the tasks into useful nested dictionary structure.
    A specific task result can then be accessed by task -> coil -> orientation.

    Args:
        results (dict): Raw results from task running.

    Returns:
        dict: Formatted results in nested dictionary structure.
    """
    formatted_results = nested_dict()
    for subdict in results:
        file_value = subdict["file"]

        task_key = CLASS_STR_TO_TASK[subdict["task"]]
        coil_key = substring_matcher(
            file_value[0] if isinstance(file_value, list) else file_value,
            EXPECTED_COILS,
        )
        orientation_key = substring_matcher(
            file_value[0] if isinstance(file_value, list) else file_value,
            EXPECTED_ORIENTATIONS,
        )
        formatted_results[task_key][coil_key][orientation_key] = subdict
    formatted_results = defaultdict_to_dict(formatted_results)
    return formatted_results
