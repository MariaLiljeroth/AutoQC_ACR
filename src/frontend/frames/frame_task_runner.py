from pathlib import Path
import tkinter as tk
from tkinter import ttk
import threading

from frontend.settings import FONT_TEXT, FONT_TITLE
from backend.run_tasks import run_tasks
from backend.dataframe_tools.dataframe_constructor import DataFrameConstructor
from backend.dataframe_tools.dataframe_formatter import DataFrameFormatter
from shared.queueing import get_queue


class FrameTaskRunner(tk.Frame):
    """Subclass of tk.Frame
    To display Hazen progress to user whilst tasks are running in the backend.

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
        GRID_DIMS (tuple): A tuple of expected grid dimensions for tk.widget.grid()
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
            out_subdirs (list[Path]): List of output subdirectories (of sorted DICOMs).
            tasks_to_run (list[str]): List of tasks to run on the input data.
        """
        super().__init__(master)

        self.in_dir = in_dir
        self.out_dir = out_dir
        self.df_path = out_dir / "Results_AutoQC_ACR.xlsx"
        self.in_subdirs = in_subdirs
        self.out_subdirs = out_subdirs
        self.tasks_to_run = tasks_to_run

        self._create_widgets()
        self._layout_widgets()
        self._configure_grid()

        self._trigger_task_running()

    def _create_widgets(self):
        """Creates widgets"""
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

        self.progress_bars = {
            "prep": ttk.Progressbar(self, mode="indeterminate"),
            "tasks": ttk.Progressbar(self, maximum=100),
        }

    def _layout_widgets(self):
        """Lays out widgets."""
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

        for key, widget in self.progress_bars.items():
            if key == "prep":
                widget.grid(row=1, column=1, sticky="nsew", pady=self.PAD_Y)
            elif key == "tasks":
                widget.grid(row=2, column=1, sticky="nsew", pady=self.PAD_Y)

    def _configure_grid(self):
        """Configures grid weights and min row/col sizes for self."""
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1, minsize=300)

    def _trigger_task_running(self):
        """Triggers task running process and starts prep progress bar.
        This bar runs until the first task is completed.
        """
        self.progress_bars["prep"].start(10)
        self.prep_bar_running = True
        args = (self.in_subdirs, self.out_subdirs, self.tasks_to_run)
        threading.Thread(target=run_tasks, args=args).start()

    def handle_event(self, event: tuple):
        """Handles events passed from main event queue (see App._check_queue).

        Args:
            event (tuple): Tuple containing unique event trigger strings.
        """
        if event[0] == "PROGRESS_BAR_UPDATE":
            # Handle progress bar related events.
            if event[1] == "TASK_RUNNING":
                # Update progress bar for task running and stop prep bar.
                if self.prep_bar_running:
                    self.progress_bars["prep"].stop()
                    self.progress_bars["prep"].config(mode="determinate")
                    self.progress_bars["prep"]["value"] = 100
                    self.prep_bar_running = False
                self.progress_bars["tasks"]["value"] += event[2]

        if event[0] == "TASK_COMPLETE":
            # Handle task completion events.
            if event[1] == "TASK_RUNNING":
                # Construct dataframe once all tasks have been run.
                dfc = DataFrameConstructor(event[2], self.df_path)
                threading.Thread(target=dfc.run).start()

            elif event[1] == "DATAFRAME_CONSTRUCTED":
                # Format dataframe once it has been constructed.
                dff = DataFrameFormatter(event[2], self.df_path)
                threading.Thread(target=dff.run).start()

            elif event[1] == "DATAFRAME_FORMATTED":
                # Quit app after dataframe has been formatted.
                get_queue().put(("QUIT_APPLICATION",))
