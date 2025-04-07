import sys
from pathlib import Path

HAZEN_DIR = Path(__file__).parents[3] / "Scottish-Medium-ACR-Analysis-Framework"
sys.path.append(str(HAZEN_DIR))

from mappings import Mappings
from cache_manager import CacheManager
from folder_manager import FolderManager
from task_runner import TaskRunner
from dataframe_creator import DataFrameCreator
from excel_formatter import ExcelFormatter
from utils import print_break


class TaskLooperRSCH:
    APP_NAME = "AutoQC_ACR"

    def run(self, tasks: list, debugCachePull: bool = False):
        self.check_for_unsupported_tasks(tasks)
        self.cache_manager = CacheManager(self.APP_NAME)

        # handling IO folder logic
        print_break(20)
        if debugCachePull:
            print("Attempting to pull previous results from cache.")
            self.folder_manager = self.cache_manager.load_cache("folder_manager")
        else:
            print("Asking user to select top-level I/O folders.")
            self.folder_manager = FolderManager()
            self.folder_manager.run()
            print_break(20)

        self.possible_keys = self.generate_keys_dict()

        # Handling results generation logic
        if debugCachePull:
            self.results = self.cache_manager.load_cache("results")
            print_break(20)
        else:
            print("Running Hazen tasks.")
            self.task_runner = TaskRunner(outer=self)
            self.results = self.task_runner.run()
            print_break(20)
            print("Storing results in cache.")
            self.cache_manager.store_cache(self.folder_manager, "folder_manager")
            self.cache_manager.store_cache(self.results, "results")
            print_break(20)

        # handling results creation logic
        self.excel_path = self.folder_manager.out_parent / "output_AutoQC_ACR.xlsx"

        self.df_creator = DataFrameCreator(outer=self)
        self.df_creator.run(self.excel_path)

        self.excel_formatter = ExcelFormatter(outer=self)
        self.excel_formatter.run(self.excel_path)
        print(f"Results successfully stored at: {self.excel_path}")
        print_break(20)

    def check_for_unsupported_tasks(self, tasks):
        key_check = [task in Mappings.TASK_TO_CLASS for task in tasks]
        if all(key_check):
            self.tasks = tasks
        else:
            err = str(
                f"Task {tasks[key_check.index(False)]} not implemented! Available tasks: "
                f"{', '.join(list(Mappings.TASK_TO_CLASS.keys()))}"
            )
            raise KeyError(err)

    def generate_keys_dict(self) -> dict:
        task_keys = [f"{task}" for task in self.tasks]
        coil_keys = self.folder_manager.get_coil_keys()
        orientation_keys = self.folder_manager.get_orientation_keys()

        possible_keys = {
            "tasks": task_keys,
            "coils": coil_keys,
            "orientations": orientation_keys,
        }
        return possible_keys


if __name__ == "__main__":
    task_looper_rsch = TaskLooperRSCH()
    task_looper_rsch.run(
        tasks=[
            "slice_thickness",
            "snr",
            "geometric_accuracy",
            "uniformity",
        ],
        debugCachePull=True,
    )
