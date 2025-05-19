import multiprocessing as mp

from backend.mappings import TASK_STR_TO_CLASS, CLASS_STR_TO_TASK
from backend.smaaf.hazenlib.utils import get_dicom_files
from backend.utils import nested_dict, defaultdict_to_dict, substring_matcher
from shared.queueing import get_queue
from shared.context import EXPECTED_ORIENTATIONS, EXPECTED_COILS


def run_tasks(task_args):
    queue = get_queue()
    task_args_with_queue = [
        (in_subdir, out_subdir, task, queue, 1 / len(task_args) * 100)
        for in_subdir, out_subdir, task in task_args
    ]

    cpu_cores = mp.cpu_count()
    pool = mp.Pool(
        processes=cpu_cores // 2 if len(task_args_with_queue) > cpu_cores // 2 else 1
    )
    try:
        results = pool.starmap(run_solo_task_on_folder, task_args_with_queue)
    finally:
        pool.close()
        pool.join()

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
    get_queue().put(("TASK_COMPLETE", "TASK_RUNNING", formatted_results))


def run_solo_task_on_folder(in_subdir, out_subdir, task, queue, perc):
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
