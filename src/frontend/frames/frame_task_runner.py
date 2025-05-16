import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

import threading

from frontend.settings import FONT_TEXT, FONT_TITLE
from backend.run_tasks import run_tasks
from backend.dataframe_tools.dataframe_constructor import DataFrameConstructor
from backend.dataframe_tools.dataframe_formatter import DataFrameFormatter
from shared.queueing import get_queue


class FrameTaskRunner(tk.Frame):

    GRID_DIMS = (4, 2)
    PAD_X = (0, 10)
    PAD_Y = (0, 15)

    def __init__(self, master, in_dir, out_dir, in_subdirs, out_subdirs, tasks_to_run):
        super().__init__(master)

        self.in_dir = in_dir
        self.out_dir = out_dir
        self.df_path = out_dir / "Results_AutoQC_ACR.xlsx"
        self.in_subdirs = in_subdirs
        self.out_subdirs = out_subdirs
        self.tasks_to_run = tasks_to_run
        self.task_args = [
            (in_subdir, out_subdir, task)
            for in_subdir, out_subdir in zip(in_subdirs, out_subdirs)
            for task in tasks_to_run
        ]

        self._create_widgets()
        self._layout_widgets()
        self._configure_grid()

        self._trigger_task_running()

    def _create_widgets(self):
        # Labels
        self.label_title = tk.Label(self, text="Task Progress", font=FONT_TITLE)
        self.label_prep_progress = tk.Label(
            self,
            text="Preparing to run:",
            font=FONT_TEXT,
            anchor="center",
        )
        self.label_tasks_progress = tk.Label(
            self,
            text="Task progress:",
            font=FONT_TEXT,
            anchor="center",
        )

        # Progress bars
        self.progress_bar_prep = ttk.Progressbar(self, mode="indeterminate")
        self.progress_bar_tasks = ttk.Progressbar(self, maximum=100)

    def _layout_widgets(self):
        # Column 0
        self.label_title.grid(
            row=0, column=0, columnspan=2, sticky="w", pady=self.PAD_Y
        )
        self.label_prep_progress.grid(
            row=1,
            column=0,
            sticky="w",
            padx=self.PAD_X,
            pady=self.PAD_Y,
        )
        self.label_tasks_progress.grid(
            row=2,
            column=0,
            sticky="w",
            padx=self.PAD_X,
            pady=self.PAD_Y,
        )

        # Column 1
        self.progress_bar_prep.grid(row=1, column=1, sticky="nsew", pady=self.PAD_Y)
        self.progress_bar_tasks.grid(row=2, column=1, sticky="nsew", pady=self.PAD_Y)

    def _configure_grid(self):
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1, minsize=300)

    def _trigger_task_running(self):
        self.progress_bar_prep.start(10)
        self.prep_bar_running = True
        threading.Thread(target=run_tasks, args=(self.task_args,)).start()

    def handle_event(self, event):
        if event[0] == "PROGRESS_BAR_UPDATE":
            if event[1] == "TASK_RUNNING":
                if self.prep_bar_running:
                    self.progress_bar_prep.stop()
                    self.progress_bar_prep.config(mode="determinate")
                    self.progress_bar_prep["value"] = 100
                    self.prep_bar_running = False
                self.progress_bar_tasks["value"] += event[2]

        if event[0] == "TASK_COMPLETE":
            if event[1] == "TASK_RUNNING":
                dfc = DataFrameConstructor(event[2], self.df_path)
                threading.Thread(target=dfc.run).start()

            elif event[1] == "DATAFRAME_CONSTRUCTED":
                dff = DataFrameFormatter(event[2], self.df_path)
                threading.Thread(target=dff.run).start()

            elif event[1] == "DATAFRAME_FORMATTED":
                get_queue().put(("QUIT_APPLICATION",))
