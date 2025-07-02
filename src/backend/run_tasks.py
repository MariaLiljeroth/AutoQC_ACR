from pathlib import Path

import multiprocessing as mp
from multiprocessing.managers import BaseProxy

from backend.mappings import TASK_STR_TO_CLASS, CLASS_STR_TO_TASK
from backend.smaaf.hazenlib.utils import get_dicom_files
from backend.utils import nested_dict, defaultdict_to_dict, substring_matcher
from backend.dev_settings import FORCE_SEQUENTIAL_PROCESSING

from shared.queueing import get_queue
from shared.context import EXPECTED_ORIENTATIONS, EXPECTED_COILS


def run_tasks(in_subdirs: list[Path], out_subdirs: list[Path], tasks_to_run: list[str]):
    """Runs all tasks on all in_subdirs, outputting to relevant out_subdirs.
    Utilises serial or multiprocessing depending on the number of jobs.
    A job is defined by a specific set of args to run a task with, e.g. input and output
    subdirs, the specific task etc. A task is the specific Hazen task e.g. SNR, Uniformity etc.

    Args:
        in_subdirs (list[Path]): List of input subdirectories.
        out_subdirs (list[Path]): List of output subdirectories.
        tasks_to_run (list[Path]): List of Hazen tasks to run.
    """

    # Formulate job_args, a list of the args to pass to individual jobs.
    # Collect the input and output subdirs, task, queue and progress bar contribution.
    queue = get_queue()
    num_jobs = len(in_subdirs) * len(tasks_to_run)
    d_prog_bar = 1 / num_jobs * 100

    job_args = [
        (in_subdir, out_subdir, task, queue, d_prog_bar)
        for in_subdir, out_subdir in zip(in_subdirs, out_subdirs)
        for task in tasks_to_run
    ]

    # Assign number of workers based on number of jobs
    jobs_per_task = len(in_subdirs)
    num_jobs_num_workers_mapping = (
        (0, jobs_per_task, 1),
        (jobs_per_task + 1, 2 * jobs_per_task, 3),
        (2 * jobs_per_task + 1, 4 * jobs_per_task, 4),
        (4 * jobs_per_task + 1, 100 * jobs_per_task, 5),
    )

    for lower, upper, num_workers in num_jobs_num_workers_mapping:
        if lower <= num_jobs <= upper:
            num_workers = (
                num_workers
                if num_workers < mp.cpu_count() and not FORCE_SEQUENTIAL_PROCESSING
                else 1
            )
            break

    # Run with either parallelism or serial processing.
    if num_workers > 1:
        with mp.Pool(processes=num_workers) as pool:
            results = pool.starmap(run_solo_task_on_folder, job_args)
    else:
        results = [run_solo_task_on_folder(*args) for args in job_args]

    # Format results in nested dictionary structure and send via queue.
    formatted_results = format_results(results)
    get_queue().put(("TASK_COMPLETE", "TASK_RUNNING", formatted_results))


def run_solo_task_on_folder(
    in_subdir: Path, out_subdir: Path, task: str, queue: BaseProxy, perc: float
) -> dict:
    """Runs a single job for a specific arg set in job_args.
    Returns the raw Hazen output for the job (which is in dict structure).

    Args:
        in_subdir (Path): Specific input subdirectory for job
        out_subdir (Path): Specific output subdirectory for job.
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

    if len(kwargs["input_data"]) != 11:
        print(
            f"Warning: For {in_subdir.name}, {task} could not be calculated because {in_subdir.name} contains an unexpected number of DICOMs. Expected 11 but received {len(kwargs["input_data"])}."
        )
        queue.put(("PROGRESS_BAR_UPDATE", "TASK_RUNNING", perc))
        return None

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
    for subdict in [r for r in results if r is not None]:
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
