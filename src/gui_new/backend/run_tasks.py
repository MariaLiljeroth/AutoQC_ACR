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
    pool = mp.Pool(processes=cpu_cores // 2)
    try:
        results = pool.starmap(run_solo_task_on_folder, task_args_with_queue)
    finally:
        pool.close()
        pool.join()

    formatted_results = nested_dict()
    for subdict in results:
        task_key = CLASS_STR_TO_TASK[subdict["task"]]
        coil_key = substring_matcher(subdict["file"], EXPECTED_COILS)
        orientation_key = substring_matcher(subdict["file"], EXPECTED_ORIENTATIONS)
        formatted_results[task_key][coil_key][orientation_key] = subdict
    formatted_results = defaultdict_to_dict(formatted_results)
    get_queue().put(("TASK_COMPLETE", "TASK_RUNNING", formatted_results))


def run_solo_task_on_folder(in_subdir, out_subdir, task, queue, perc):
    task_obj = TASK_STR_TO_CLASS[task](
        input_data=get_dicom_files(in_subdir.resolve()),
        report_dir=out_subdir.resolve(),
        report=True,
        MediumACRPhantom=True,
    )
    result = task_obj.run()
    queue.put(("PROGRESS_BAR_UPDATE", "TASK_RUNNING", perc))
    return result
