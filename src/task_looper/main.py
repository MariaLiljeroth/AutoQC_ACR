import sys
from pathlib import Path

# CONFIGURE SYS.PATH FOR IMPORTS
sys.path.append(
    str(
        (Path(__file__).parents[1] / "Scottish-Medium-ACR-Analysis-Framework").resolve()
    )
)

import config
import settings
import logging

from cache_manager import CacheManager
from folder_manager import FolderManager
from task_runner import TaskRunner
from dataframe_creator import DataFrameCreator
from excel_formatter import ExcelFormatter


class TaskLooperRSCH:
    APP_NAME = "AutoQC_ACR"

    def run(self):
        self.check_for_unsupported_tasks()
        self.cache_manager = CacheManager(self.APP_NAME)

        if settings.PULL_FROM_CACHE:
            self.folder_manager = self.cache_manager.load_cache("folder_manager")
            self.results = self.cache_manager.load_cache("results")
        else:
            self.folder_manager = FolderManager()
            self.folder_manager.run()

            self.task_runner = TaskRunner(outer=self)
            self.results = self.task_runner.run()

            self.cache_manager.store_cache(self.folder_manager, "folder_manager")
            self.cache_manager.store_cache(self.results, "results")

        # WORKS UP TO HERE - JUST NEED TO FIX BELOW TO ACCOUNT FOR SETTINGS.PY AND CONFIG.PY
        # handling results creation logic
        self.excel_path = self.folder_manager.out_dir / "output_AutoQC_ACR.xlsx"

        self.df_creator = DataFrameCreator(outer=self)
        self.df_creator.run(self.excel_path)

        self.excel_formatter = ExcelFormatter(outer=self)
        self.excel_formatter.run(self.excel_path)
        logging.info(f"Results successfully stored at: {self.excel_path}")

    def check_for_unsupported_tasks(self):
        key_check = [task in config.IMPLEMENTED_TASKS for task in settings.TASKS_TO_RUN]
        if not all(key_check):
            err = str(
                f"Task {settings.TASKS_TO_RUN[key_check.index(False)]} not implemented! Available tasks: "
                f"{', '.join(config.IMPLEMENTED_TASKS)}"
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
    task_looper_rsch.run()
