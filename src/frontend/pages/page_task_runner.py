"""
page_task_runner.py

This file defines the PageTaskRunner class, which is a tk.Frame subclass that shows the progress of
the running of hazen tasks in the backend. Is swapped in from configuration page and is currently the final page.

Written by Nathan Crossley 2025

"""

import threading
import pydicom

import tkinter as tk
from tkinter import ttk

from src.frontend.app_state import AppState
from src.frontend.settings import FONT_TEXT, FONT_TITLE, PAD_LARGE, PAD_MEDIUM

from src.backend.run_all_jobs import run_all_jobs
from src.backend.log_tools.log_constructor import LogConstructor
from src.backend.report_tools.report_generator import ReportGenerator

from src.shared.queueing import get_queue, QueueTrigger


class PageTaskRunner(tk.Frame):
    """Subclass of tk.Frame used to display to user the progress
    of Hazen tasks running in the backend.
    """

    def __init__(self, master: tk.Tk, app_state: AppState):
        """Initialises PageTaskRunner.

        Creates, configures and lays out widgets within self.
        Finally, executes job running pipeline.

        Args:
            master (tk.Tk): Root window within which self is placed.
            app_state (AppState): class for sharing app state within frontend
        """
        super().__init__(master)

        # save reference to shared app state class so can access attributes modified in other pages
        self.app_state = app_state

        # add path to log to app state
        self.app_state.log_path = self.app_state.out_dir / "Results_AutoQC_ACR.xlsx"

        # create widgets within self
        self._create_widgets()
        self._layout_widgets()
        self._configure_grid()

        # triggers job running process after page is initialised
        self._trigger_job_running()

    def _create_widgets(self):
        """Creates widgets within self. Individual functions
        are called for different types of widget."""
        self._create_labels()
        self._create_progress_bars()

    def _create_labels(self):
        """Create all labels required for self"""

        # create label for page title
        self.label_title = tk.Label(self, text="MR Job Progress", font=FONT_TITLE)

        # create label to indicate what prep progress bar represents
        self.label_prep_progress = tk.Label(
            self,
            text="Preparing workers:",
            font=FONT_TEXT,
            anchor="center",
        )

        # create label to indicate what job progress bar represents
        self.label_job_progress = tk.Label(
            self,
            text="Job progress:",
            font=FONT_TEXT,
            anchor="center",
        )

    def _create_progress_bars(self):
        """Create all progress bars required for self"""

        # create progress bar for multiprocessing prep
        self.progress_bar_prep = ttk.Progressbar(self, mode="indeterminate")

        # create progress bar for jobs completed
        self.progress_bar_jobs = ttk.Progressbar(self, maximum=100)

    def _layout_widgets(self):
        """Lays out widgets within self. Individual
        functions are called for different widget types"""
        self._layout_labels()
        self._layout_progress_bars()

    def _layout_labels(self):
        """Lays out all labels within self"""
        self.label_title.grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, PAD_LARGE)
        )
        self.label_prep_progress.grid(
            row=1,
            column=0,
            sticky="w",
            padx=(0, PAD_MEDIUM),
            pady=(0, PAD_LARGE),
        )
        self.label_job_progress.grid(
            row=2,
            column=0,
            sticky="w",
            padx=(0, PAD_MEDIUM),
            pady=(0, PAD_LARGE),
        )

    def _layout_progress_bars(self):
        """Lays out all progress bars within self"""
        self.progress_bar_prep.grid(row=1, column=1, sticky="nsew", pady=(0, PAD_LARGE))
        self.progress_bar_jobs.grid(row=2, column=1, sticky="nsew", pady=(0, PAD_LARGE))

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

    def _trigger_job_running(self):
        """Triggers running of Hazen jobs in the backend and
        starts indeterminate multiprocessing preparation progress bar.
        This bar runs until the first job is completed.
        """

        # start progress bar to signal to user that multiprocessing workers being set up
        self.progress_bar_prep.start(10)

        # track the fact that the prep bar is running in a bool, for use later in turning it off
        self.prep_bar_running = True

        # collect args to pass to backend for running of Hazen jobs
        args = (
            self.app_state.in_subdirs,
            self.app_state.out_subdirs,
            self.app_state.tasks_to_run,
        )

        # create a thread within which Hazen jobs can run without blocking gui (main thread)
        threading.Thread(target=run_all_jobs, args=args).start()

    def handle_task_request(self, trigger: QueueTrigger):
        """Handles triggers passed through from main event
        queue (see App._check_queue).

        Args:
            trigger (QueueTrigger): Queue trigger relating to this page.
        """

        # Handle trigger to update progress bar after job completed
        if trigger.ID == "PROGBAR_UPDATE_JOB_COMPLETED":

            # If prep bar still running, stop it as setup completed now
            if self.prep_bar_running:
                self.progress_bar_prep.stop()
                self.progress_bar_prep.config(mode="determinate")
                self.progress_bar_prep["value"] = 100
                self.prep_bar_running = False

            # add progress to jobs progress bar due to the completion of the task
            self.progress_bar_jobs["value"] += trigger.data

        # Handle log construction trigger (from completion of job running)
        elif trigger.ID == "PRESENT_RESULTS":
            self.app_state.results = trigger.data

            # create an instance of class for constructing log
            lc = LogConstructor(self.app_state.results, self.app_state.log_path)

            # start the construction process in a separate thread to avoid blocking gui (main thread)
            thread_lc = threading.Thread(target=lc.run)
            thread_lc.start()

            # get field strength
            test_subdir = self.app_state.in_subdirs[0]
            test_files = list(test_subdir.glob("*"))
            test_dcm_path = [x for x in test_files if x.is_file()][0]
            field_strength = pydicom.dcmread(test_dcm_path).get("MagneticFieldStrength")

            if isinstance(field_strength, str):
                field_strength = float(field_strength)

            # also construct report
            rg = ReportGenerator(
                self.app_state.results,
                self.app_state.baselines,
                field_strength,
                self.app_state.out_dir,
            )
            thread_rg = threading.Thread(target=rg.run)
            thread_rg.start()

            # wait till threads have joined and send trigger to quit app
            timeout = 10
            thread_lc.join(timeout)
            thread_rg.join(timeout)

            if thread_lc.is_alive():
                print("Warning: Log Construction thread still alive after timeout!")
            if thread_rg.is_alive():
                print("Warning: Report Generator thread still alive after timeout!")

            get_queue().put(QueueTrigger("QUIT_APPLICATION"))
