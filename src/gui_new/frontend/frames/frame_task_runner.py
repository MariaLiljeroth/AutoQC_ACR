import tkinter as tk
from tkinter import ttk

import threading

from shared.context import AVAILABLE_TASKS
from frontend.settings import FONT_TEXT, FONT_TITLE
from backend.run_tasks import run_tasks


class FrameTaskRunner(tk.Frame):

    GRID_DIMS = (4, 2)
    PAD_X = (0, 10)
    PAD_Y = (0, 15)

    def __init__(self, master, in_dir, out_dir, in_subdirs, out_subdirs, tasks_to_run):
        super().__init__(master)

        self.in_dir = in_dir
        self.out_dir = out_dir
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
        if isinstance(event, tuple) and event[0] == "UPDATE: PROGRESS BAR TASKS":
            if self.prep_bar_running:
                self.progress_bar_prep.stop()
                self.progress_bar_prep.config(mode="determinate")
                self.progress_bar_prep["value"] = 100
                self.prep_bar_running = False
            self.progress_bar_tasks["value"] += event[1]
