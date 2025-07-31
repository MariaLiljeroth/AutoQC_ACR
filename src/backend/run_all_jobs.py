"""
run_all_jobs.py

This script defines functions that are used to run all specified Hazen jobs in the backend.
Multiprocessing is utilised to vastly speed up task running, based on the number of CPU cores
compared to the number of Hazen jobs requested. The results are sent to the frontend via
the global queue.

Written by Nathan Crossley 2025

"""

from pathlib import Path
import pydicom

import multiprocessing as mp
from multiprocessing.managers import BaseProxy

from src.backend.mappings import TASK_STR_TO_CLASS, CLASS_STR_TO_TASK
from src.backend.hazen.hazenlib.utils import get_dicom_files
from src.backend.utils import nested_dict, defaultdict_to_dict, substring_matcher
from dev_settings import FORCE_SEQUENTIAL_PROCESSING

from src.shared.queueing import get_queue, QueueTrigger
from src.shared.context import (
    EXPECTED_ORIENTATIONS,
    EXPECTED_COILS,
    IMPLEMENTED_MANUFACTURERS,
)


def run_all_jobs(
    in_subdirs: list[Path], out_subdirs: list[Path], tasks_to_run: list[str]
):
    """Runs all specified tasks on all input subdirectories, saving plots to the corresponding.
    output subdirectories. Utilises serial or multiprocessing depending on the number of CPU
    cores available and how many Hazen jobs are requested.

    For clarity, a "job" is defined by a specific set of args to run a task with, e.g. input and output
    subdirs, the specific task etc. A task is the specific Hazen task e.g. SNR, Uniformity etc.

    Args:
        in_subdirs (list[Path]): List of input subdirectories.
        out_subdirs (list[Path]): List of output subdirectories.
        tasks_to_run (list[str]): List of Hazen tasks to run.
    """

    # Formulate job_args, a list of the args to pass to individual jobs.
    # Collect the input and output subdirs, task, queue and progress bar contribution.

    # get multiprocessing queue
    queue = get_queue()

    # calculate the number of jobs requested
    num_jobs = len(in_subdirs) * len(tasks_to_run)

    # calculate the progress bar change associated with the completion of a single job.
    d_prog_bar = 1 / num_jobs * 100

    # zip up arguments for individual jobs.
    job_args = [
        (in_subdir, out_subdir, task, queue, d_prog_bar)
        for in_subdir, out_subdir in zip(in_subdirs, out_subdirs)
        for task in tasks_to_run
    ]

    # define relationship between number of jobs and number of multiprocessing workers assigned
    jobs_per_task = len(in_subdirs)
    num_jobs_num_workers_mapping = (
        (0, jobs_per_task, 1),
        (jobs_per_task + 1, 2 * jobs_per_task, 3),
        (2 * jobs_per_task + 1, 4 * jobs_per_task, 4),
        (4 * jobs_per_task + 1, 100 * jobs_per_task, 5),
    )

    # apply above relationship to work out how many workers will be assigned for specific workload
    for lower, upper, num_workers in num_jobs_num_workers_mapping:
        if lower <= num_jobs <= upper:
            num_workers = (
                num_workers
                if num_workers < mp.cpu_count() and not FORCE_SEQUENTIAL_PROCESSING
                else 1
            )
            break

    # Run with either parallelism or serial processing based on number of workers
    if num_workers > 1:

        # create multiprocessing pool, running job running func with job arguments
        with mp.Pool(processes=num_workers) as pool:
            results = pool.starmap(run_job, job_args)

    else:
        # run jobs serially (no multiprocessing)
        results = [run_job(*args) for args in job_args]

    # Format results in nested dictionary structure (for clarity and ease of dataframe construction)
    formatted_results = format_results(results)

    # signal to queue that task running process has been completed and should now construct log and report
    get_queue().put(QueueTrigger("PRESENT_RESULTS", formatted_results))


def run_job(
    in_subdir: Path, out_subdir: Path, task: str, queue: BaseProxy, perc: float
) -> dict:
    """Runs a single job for a specific set of input args from job_args.
    Returns the raw Hazen output for the job (which is in dict structure).
    Note that the queue must be passed in manually as an arg to allow this func
    to be pickleable for multiprocessing pool.

    Args:
        in_subdir (Path): Specific input subdirectory for job
        out_subdir (Path): Specific output subdirectory for job.
        task (str): Task to run.
        queue (BaseProxy): Queue to send progress updates to.
        perc (float): Float value to add to current state of progress bar.

    Returns:
        dict: Raw Hazen result for specific task_args tuple.
    """

    # Collect kwargs that need to be passed to custom Hazen task class
    kwargs = {
        "input_data": get_dicom_files(in_subdir.resolve()),
        "report_dir": out_subdir.resolve(),
        "report": True,
        "MediumACRPhantom": True,
    }

    # catch case (before job runs) where input dicom subdirectory has a number of dcms != 11, because otherwise
    # Hazen will produce inaccurate results. print warning message and still increment progress bar else
    # visuals will be messed up
    if len(kwargs["input_data"]) != 11:
        print(
            f"Warning: For {in_subdir.name}, {task} could not be calculated because {in_subdir.name} contains an unexpected number of DICOMs. Expected 11 but received {len(kwargs["input_data"])}."
        )
        queue.put(QueueTrigger("PROGBAR_UPDATE_JOB_COMPLETED", perc))
        return None

    # if Philips scanner and SNR task called, pass helper data set to task class through subtract kwarg
    helper_data_set = in_subdir / "helper_data_set"
    manufacturer_tag = pydicom.dcmread(kwargs["input_data"][0]).get("Manufacturer")
    manufacturer = substring_matcher(manufacturer_tag, IMPLEMENTED_MANUFACTURERS)
    if task == "SNR" and manufacturer == "Philips":
        kwargs["subtract"] = helper_data_set

    # map task string to associated class and instantiate, passing kwargs
    task_obj = TASK_STR_TO_CLASS[task](**kwargs)

    # run job to get result
    result = task_obj.run()

    # Update job running progress bar
    queue.put(QueueTrigger("PROGBAR_UPDATE_JOB_COMPLETED", perc))

    return result


def format_results(results: dict) -> dict:
    """Formats the raw results from running all Hazen jobs into useful nested dictionary structure.
    A specific job result can then be accessed by calling [task][coil][orientation] on the formatted dict.

    Args:
        results (dict): Raw results from Hazen job running.

    Returns:
        dict: Formatted results in nested dictionary structure.
    """

    # Get instance of nested dict
    formatted_results = nested_dict()

    # Iterate over subdicts i.e. those generated by individual jobs
    for subdict in [r for r in results if r is not None]:

        # get "file" description, which has original input subdir as a substring
        file_value = subdict["file"]

        # get task name by mapping task class back to string
        task_key = CLASS_STR_TO_TASK[subdict["task"]]

        # get best matching coil by matching file description with expected coils
        coil_key = substring_matcher(
            file_value[0] if isinstance(file_value, list) else file_value,
            EXPECTED_COILS,
        )

        # get best matching orientation by matching file description with expected orientations
        orientation_key = substring_matcher(
            file_value[0] if isinstance(file_value, list) else file_value,
            EXPECTED_ORIENTATIONS,
        )

        # append results subdict to formatted results nested dict under organised structure
        formatted_results[task_key][coil_key][orientation_key] = subdict

    # convert formatted results back to normal dict from nested default_dict
    formatted_results = defaultdict_to_dict(formatted_results)

    return formatted_results
