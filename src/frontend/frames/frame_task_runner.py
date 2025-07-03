"""
frame_task_runner.py

This file defines the FrameTaskRunner class, which is a tk.Frame subclass that shows the progress of
the running of hazen tasks in the backend. Is swapped in from configuration frame and is currently the final frame.

Written by Nathan Crossley 2025

"""

from pathlib import Path
import threading

import tkinter as tk
from tkinter import ttk

from frontend.settings import FONT_TEXT, FONT_TITLE

from backend.run_tasks import run_tasks
from backend.dataframe_tools.dataframe_constructor import DataFrameConstructor
from backend.dataframe_tools.dataframe_formatter import DataFrameFormatter

from shared.queueing import get_queue


class FrameTaskRunner(tk.Frame):
    """Subclass of tk.Frame
    Used to display to user the progress of Hazen tasks running in the backend.

    Instance attributes:
        in_dir (Path): Parent directory for input data.
        out_dir (Path): Parent directory for output data.
        df_path (Path): Path to the output Excel dataframe.
        in_subdirs (list[Path]): List of input subdirectories (of sorted DICOMs).
        out_subdirs (list[Path]): List of output subdirectories (of sorted DICOMs).
        tasks_to_run (list[str]): List of tasks to run on the input data.
        labels (dict[str, tk.Label]): Dictionary of tk.Label widgets.
        progress_bars (dict[str, ttk.Progressbar]): Dictionary of tk.Progressbar widgets.
        prep_bar_running (bool): Flag to indicate if the preparation progress bar is running.


    Class attributes:
        GRID_DIMS (tuple): A tuple of expected grid dimensions for grid of self.
        PAD_X (tuple): Standardised x-padding between grid columns.
        PAD_Y (tuple): Standardised y-padding between grid rows.

    """

    GRID_DIMS = (4, 2)
    PAD_X = (0, 10)
    PAD_Y = (0, 15)

    def __init__(
        self,
        master: tk.Tk,
        in_dir: Path,
        out_dir: Path,
        in_subdirs: list[Path],
        out_subdirs: list[Path],
        tasks_to_run: list[str],
    ):
        """Initialises FrameTaskRunner.

        Args:
            master (tk.Tk): Root window of the application.
            in_dir (Path): Parent directory for input data.
            out_dir (Path): Parent directory for output data.
            in_subdirs (list[Path]): List of input subdirectories (of sorted DICOMs).
            out_subdirs (list[Path]): List of output subdirectories (for results).
            tasks_to_run (list[str]): List of tasks to run on the input data.
        """
        super().__init__(master)

        # save args as instance attributes for convenience
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.df_path = out_dir / "Results_AutoQC_ACR.xlsx"
        self.in_subdirs = in_subdirs
        self.out_subdirs = out_subdirs
        self.tasks_to_run = tasks_to_run

        # create widgets within self
        self._create_widgets()
        self._layout_widgets()
        self._configure_grid()

        # triggers task running process after frame is initialised
        self._trigger_task_running()

    def _create_widgets(self):
        """Creates widgets within self"""
        # Dict containing string-id and widget pairs for all labels within self.
        # Labels are used to convey information about what the progress bars represent.
        self.labels = {
            "title": tk.Label(self, text="Task Progress", font=FONT_TITLE),
            "prep_progress": tk.Label(
                self,
                text="Preparing workers:",
                font=FONT_TEXT,
                anchor="center",
            ),
            "tasks_progress": tk.Label(
                self,
                text="Task progress:",
                font=FONT_TEXT,
                anchor="center",
            ),
        }

        # Dict containing string-id and widget pairs for all progress bar within self.
        # Progress bars used to show progress of Hazen tasks in backend.
        self.progress_bars = {
            "prep": ttk.Progressbar(self, mode="indeterminate"),
            "tasks": ttk.Progressbar(self, maximum=100),
        }

    def _layout_widgets(self):
        """Lays out widgets within self."""

        # position all labels within grid of self
        for key, widget in self.labels.items():
            if key == "title":
                widget.grid(row=0, column=0, columnspan=2, sticky="w", pady=self.PAD_Y)
            elif key == "prep_progress":
                widget.grid(
                    row=1,
                    column=0,
                    sticky="w",
                    padx=self.PAD_X,
                    pady=self.PAD_Y,
                )
            elif key == "tasks_progress":
                widget.grid(
                    row=2,
                    column=0,
                    sticky="w",
                    padx=self.PAD_X,
                    pady=self.PAD_Y,
                )

        # position all progress bars within grid of self
        for key, widget in self.progress_bars.items():
            if key == "prep":
                widget.grid(row=1, column=1, sticky="nsew", pady=self.PAD_Y)
            elif key == "tasks":
                widget.grid(row=2, column=1, sticky="nsew", pady=self.PAD_Y)

    def _configure_grid(self):
        """Configures weights and minimum sizes for columns and
        rows within grid of self."""

        # set row weights
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)

        # set column weights and minimum sizes
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1, minsize=300)

    def _trigger_task_running(self):
        """Triggers running of Hazen tasks in the backend and
        starts indeterminate multiprocessing preparation progress bar.
        This bar runs until the first task is completed.
        """

        # start progress bar to signal to user that multiprocessing workers being set up
        self.progress_bars["prep"].start(10)

        # track the fact that the prep bar is running in a bool, for use later in turning it off
        self.prep_bar_running = True

        # collect args to pass to backend for running of Hazen tasks
        args = (self.in_subdirs, self.out_subdirs, self.tasks_to_run)

        # create a thread within which Hazen tasks can run without blocking gui (main thread)
        threading.Thread(target=run_tasks, args=args).start()

    def handle_event(self, event: tuple):
        """Handles events passed through from main event
        queue (see App._check_queue).

        Args:
            event (tuple): Tuple containing unique event trigger strings
                (see shared/queueing.py).
        """

        # Handle events relating to a progress bar update
        if event[0] == "PROGRESS_BAR_UPDATE":

            # Triggers progress bar update caused by a completed Hazen task
            if event[1] == "TASK_RUNNING":

                # If prep bar still running, stop it as setup completed now
                if self.prep_bar_running:
                    self.progress_bars["prep"].stop()
                    self.progress_bars["prep"].config(mode="determinate")
                    self.progress_bars["prep"]["value"] = 100
                    self.prep_bar_running = False

                # add progress to tasks progress bar due to the completion of the task
                self.progress_bars["tasks"]["value"] += event[2]

        # Handle events relating to a specific "task" being complete
        if event[0] == "TASK_COMPLETE":

            # Handle completion of the running of all Hazen tasks
            if event[1] == "TASK_RUNNING":

                # create an instance of class for constructing output dataframe
                dfc = DataFrameConstructor(event[2], self.df_path)

                # start the construction process in a separate thread to avoid blocking gui (main thread)
                threading.Thread(target=dfc.run).start()

            # Handle completion of dataframe construction
            elif event[1] == "DATAFRAME_CONSTRUCTED":

                # create an instance of class for constructing output dataframe
                dff = DataFrameFormatter(event[2], self.df_path)

                # start the formatting process in a separate thread to avoid blocking gui (main thread)
                threading.Thread(target=dff.run).start()

            # Handle completion of dataframe formatting
            elif event[1] == "DATAFRAME_FORMATTED":

                # Trigger the quitting of whole application as everything is now done
                get_queue().put(("QUIT_APPLICATION",))
