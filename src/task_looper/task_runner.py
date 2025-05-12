from pathlib import Path
import logging

import config
from settings import TASKS_TO_RUN
from utils import nested_dict, defaultdict_to_dict, substring_matcher
from hazenlib.utils import get_dicom_files


class TaskRunner:
    def __init__(self, outer: "TaskLooperRSCH"):
        self.outer = outer

    def run(self):
        logging.info("Running Hazen tasks...")
        self.results = nested_dict()
        self.populate_results_dict()
        self.results = defaultdict_to_dict(self.results)

        return self.results

    def populate_results_dict(self):
        # get raw results output from hazen code
        raw_results = [
            self._run_tasks_on_folder(in_subdir, out_subdir, tasks=TASKS_TO_RUN)
            for (in_subdir, out_subdir) in zip(
                self.outer.folder_manager.in_subdirs,
                self.outer.folder_manager.out_subdirs,
            )
        ]

        # populate results dict (organised results)
        for outer_list in raw_results:
            for inner_dict in outer_list:
                task_key = config.CLASS_STR_TO_TASK[inner_dict["task"]]
                coil_key = substring_matcher(inner_dict["file"], config.EXPECTED_COILS)
                orientation_key = substring_matcher(
                    inner_dict["file"], config.EXPECTED_ORIENTATIONS
                )
                self.results[task_key][coil_key][orientation_key] = inner_dict

    @staticmethod
    def _run_tasks_on_folder(
        in_subdir: Path, out_subdir: Path, tasks: list[str]
    ) -> list[dict]:
        task_objs = [
            config.TASK_TO_CLASS[task](
                input_data=get_dicom_files(in_subdir.resolve()),
                report_dir=out_subdir.resolve(),
                report=True,
                MediumACRPhantom=True,
            )
            for task in tasks
        ]
        results_for_folder = [taskobj.run() for taskobj in task_objs]
        return results_for_folder
