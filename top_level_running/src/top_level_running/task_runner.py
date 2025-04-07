import os

from mappings import Mappings
from str_utils import StrUtils
from utils import nested_dict, defaultdict_to_dict
from hazenlib.utils import get_dicom_files


class TaskRunner:
    def __init__(self, outer: "TaskLooperRSCH"):
        self.outer = outer

    def run(self):

        self.results = nested_dict()
        self.populate_results_dict()
        self.results = defaultdict_to_dict(self.results)

        return self.results

    def populate_results_dict(self):
        # get raw results output from hazen code
        raw_results = [
            self._run_tasks_on_folder(in_dir, out_dir, tasks=self.outer.tasks)
            for (in_dir, out_dir) in zip(
                self.outer.folder_manager.in_children,
                self.outer.folder_manager.out_children,
            )
        ]

        def get_matching_coil(file_name: str):
            for coil in self.outer.possible_keys["coils"]:
                if coil in file_name:
                    return coil
            raise ValueError(f"No matching coil found for file: {file_name}")

        def get_matching_orientation(file_name: str):
            for orientation in self.outer.possible_keys["orientations"]:
                if orientation in file_name:
                    return orientation
            raise ValueError(f"No matching orientation found for file: {file_name}")

        # populate results dict (organised results)
        for outer_list in raw_results:
            for inner_dict in outer_list:
                task_key = Mappings.CLASS_STR_TO_TASK[inner_dict["task"]]
                coil_key = get_matching_coil(inner_dict["file"])
                orientation_key = get_matching_orientation(inner_dict["file"])
                self.results[task_key][coil_key][orientation_key] = inner_dict

    @staticmethod
    def _run_tasks_on_folder(in_dir: str, out_dir: str, tasks: list) -> list[dict]:
        input_data = get_dicom_files(os.path.join(in_dir))
        task_objs = [
            Mappings.TASK_TO_CLASS[task](
                input_data=input_data,
                report_dir=out_dir,
                report=True,
                MediumACRPhantom=True,
            )
            for task in tasks
        ]
        results_folder = [taskobj.run() for taskobj in task_objs]
        return results_folder
