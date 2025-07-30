"""
page_config.py

This file defines the PageConfig tk.Frame subclass, whose purpose is to allow the user to configure AutoQC_ACR settings before running it.
This page is the first page to appear within the App instance, and is swapped out to PageTaskRunner instance once hazen tasks begin running.

Written by Nathan Crossley 2025

"""

from pathlib import Path
import threading

import tkinter as tk
from tkinter import filedialog

from src.shared.context import AVAILABLE_TASKS, EXPECTED_COILS
from src.shared.queueing import get_queue

from src.frontend.settings import FONT_TEXT, FONT_TITLE, PAD_MEDIUM, PAD_LARGE
from src.frontend.widgets.modal_progress_bar import ModalProgressBar
from src.frontend.widgets.scrollable_listbox import ScrollableListbox
from src.frontend.widgets.checkbutton_panel import CheckbuttonPanel
from src.frontend.widgets.simple_table import SimpleTable
from src.frontend.tk_utils import populate_entry_widget

from src.backend.dcm_sorter import DcmSorter
from src.backend.configuration_tests import file_structure_problems_exist


class PageConfig(tk.Frame):
    def __init__(self, master: tk.Tk):
        """Initialises PageConfig instance. Creates, configures
        and lays out widgets within self.

        Args:
            master (tk.Tk): Root window from on which self is placed.
        """
        super().__init__(master)

        # Create, configure and layout widgets within self.
        self._create_widgets()
        self._layout_widgets()

        # Configure grid dimensions within self.
        self._configure_grid()

        # Set to None as no modal progress bar exists at initialisation.
        self.modal_progress = None

    def _create_widgets(self):
        """Creates widgets within self."""
        self._create_labels()
        self._create_entries()
        self._create_scrollable_listboxes()
        self._create_checkbutton_panels()
        self._create_buttons()
        self._create_tables()

    def _create_labels(self):
        self.label_title = tk.Label(
            self,
            text="Configuration Settings",
            font=FONT_TITLE,
            anchor="w",
        )
        self.label_in_dir = tk.Label(self, text="Input directory:", font=FONT_TEXT)
        self.label_out_dir = tk.Label(self, text="Output directory:", font=FONT_TEXT)
        self.label_select_subdirs = tk.Label(
            self,
            text="Select subdirectories\nto process:",
            font=FONT_TEXT,
            justify=tk.LEFT,
        )
        self.label_select_tasks = tk.Label(
            self, text="Select tasks to run:", font=FONT_TEXT
        )
        self.label_input_baselines = tk.Label(
            self, text="Input baselines:", font=FONT_TEXT
        )

    def _create_entries(self):
        self.entry_in_dir = tk.Entry(self, font=FONT_TEXT, bd=1, relief=tk.SOLID)
        self.entry_out_dir = tk.Entry(self, font=FONT_TEXT, bd=1, relief=tk.SOLID)

    def _create_scrollable_listboxes(self):
        self.scrollable_listbox_subdirs = ScrollableListbox(self)

    def _create_checkbutton_panels(self):
        self.checkbutton_panel_tasks = CheckbuttonPanel(
            self, checkbutton_names=AVAILABLE_TASKS
        )

    def _create_buttons(self):
        self.button_browse_in_dir = tk.Button(
            self,
            text="Browse",
            font=FONT_TEXT,
            command=self._browse_in_dir,
            bd=1,
            relief=tk.SOLID,
        )
        self.button_browse_out_dir = tk.Button(
            self,
            text="Browse",
            font=FONT_TEXT,
            command=self._browse_out_dir,
            bd=1,
            relief=tk.SOLID,
        )
        self.button_run = tk.Button(
            self,
            text="Run AutoQC_ACR",
            font=FONT_TEXT,
            command=self._trigger_task_running,
            bd=1,
            relief=tk.SOLID,
        )

    def _create_tables(self):
        self.table_baselines = SimpleTable(
            self,
            [AVAILABLE_TASKS[1]],
            EXPECTED_COILS,
            border_pad=PAD_LARGE,
            bd=1,
            relief=tk.SOLID,
        )

    def _layout_widgets(self):
        self._layout_labels()
        self._layout_entries()
        self._layout_scrollable_listboxes()
        self._layout_checkbutton_panels()
        self._layout_buttons()
        self._layout_tables()

    def _layout_labels(self):
        self.label_title.grid(
            row=0,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(0, PAD_LARGE),
        )
        self.label_in_dir.grid(
            row=1, column=0, sticky="w", padx=(0, PAD_MEDIUM), pady=(0, PAD_LARGE)
        )
        self.label_out_dir.grid(
            row=2, column=0, sticky="w", padx=(0, PAD_MEDIUM), pady=(0, PAD_LARGE)
        )
        self.label_select_subdirs.grid(
            row=3, column=0, sticky="nw", padx=(0, PAD_MEDIUM), pady=(0, PAD_LARGE)
        )
        self.label_select_tasks.grid(
            row=4, column=0, sticky="nw", padx=(0, PAD_MEDIUM), pady=(0, PAD_LARGE)
        )
        self.label_input_baselines.grid(
            row=5, column=0, sticky="nw", padx=(0, PAD_MEDIUM), pady=(0, PAD_LARGE)
        )

    def _layout_entries(self):
        self.entry_in_dir.grid(
            row=1, column=1, sticky="ew", padx=(0, PAD_MEDIUM), pady=(0, PAD_LARGE)
        )
        self.entry_out_dir.grid(
            row=2, column=1, sticky="ew", padx=(0, PAD_MEDIUM), pady=(0, PAD_LARGE)
        )

    def _layout_scrollable_listboxes(self):
        self.scrollable_listbox_subdirs.grid(
            row=3, column=1, sticky="ew", padx=(0, PAD_MEDIUM), pady=(0, PAD_LARGE)
        )

    def _layout_checkbutton_panels(self):
        self.checkbutton_panel_tasks.grid(
            row=4, column=1, sticky="w", padx=(0, PAD_MEDIUM), pady=(0, PAD_LARGE)
        )

    def _layout_buttons(self):
        self.button_browse_in_dir.grid(row=1, column=2, sticky="w", pady=(0, PAD_LARGE))
        self.button_browse_out_dir.grid(
            row=2, column=2, sticky="w", pady=(0, PAD_LARGE)
        )
        self.button_run.grid(row=6, column=0, columnspan=3, sticky="ew", padx=50)

    def _layout_tables(self):
        self.table_baselines.grid(
            row=5, column=1, padx=(0, PAD_MEDIUM), pady=(0, PAD_LARGE)
        )

    def _configure_grid(self):
        """Configures weights and minimum sizes for columns and
        rows within grid of self."""

        # set column weights and minimum sizes
        self.columnconfigure(0, weight=0, minsize=0)
        self.columnconfigure(1, weight=1, minsize=400)
        self.columnconfigure(2, weight=0, minsize=0)

        # set row weights and minimum sizes
        self.rowconfigure(0, weight=0, minsize=0)
        self.rowconfigure(1, weight=0, minsize=0)
        self.rowconfigure(2, weight=0, minsize=50)
        self.rowconfigure(3, weight=1, minsize=0)
        self.rowconfigure(4, weight=0, minsize=0)
        self.rowconfigure(5, weight=0, minsize=0)
        self.rowconfigure(6, weight=0, minsize=0)

    def _browse_in_dir(self):
        """Asks user to choose an input directory (activated by a button).
        This should be a directory full of sorted or unsorted DICOMs.
        Populates in_dir entry with the users choice, populates out_dir
        entry with default output dir and starts a thread to execute
        DICOM sorting process.
        """

        # get user to select input directory through basic gui
        self.in_dir = Path(filedialog.askdirectory(title="Select Input Directory"))

        if self.in_dir:
            # populates in_dir entry widget with selected input directory
            populate_entry_widget(self.entry_in_dir, str(self.in_dir.resolve()))

            # populates out_dir entry widget with default output directory based on input directory
            populate_entry_widget(
                self.entry_out_dir,
                str((self.in_dir.parent / "AutoQC_ACR_Output").resolve()),
            )

            # Instantiates a modal window for progress bar
            self.modal_progress = ModalProgressBar(self, "Checking for DICOMs")

            # Instantiates an instance of class for sorting DICOMs
            ds = DcmSorter(self.in_dir)

            # starts running DICOM sorting process in separate thread to prevent blocking gui
            threading.Thread(target=ds.run).start()

    def _browse_out_dir(self):
        """Asks user to choose an output directory for results (activated through button).
        Populates self.entry_out_dir with the users choice
        """

        # gets user to select output directory through gui
        out_dir = Path(filedialog.askdirectory(title="Select Output Directory"))

        if out_dir:
            # populate output directory entry widget with user's choice
            populate_entry_widget(self.entry_out_dir, str(out_dir.resolve()))

    def _trigger_task_running(self):
        config_settings = self._read_config_settings()
        valid = self._config_settings_valid(*config_settings)

        # signal a switch to taskrunner page and pass args
        if valid:
            get_queue().put(("SWITCH_PAGE", "TASKRUNNER", config_settings))
        else:
            return

    def _read_config_settings(self):
        """Pulls configuration settings from widgets after performing relevant
        validation checks. Sends a message to global queue to switch page to
        'TASKRUNNER', also passing configuration settings.
        """

        # get input and output directories from relevant entry widgets
        in_dir = Path(self.entry_in_dir.get())
        out_dir = Path(self.entry_out_dir.get())

        # construct input subdirectories from in_dir and relevant listbox
        in_subdirs = [
            in_dir / selection
            for selection in self.scrollable_listbox_subdirs.get_selected_items()
        ]

        # infer output subdirectories from input subdirectories
        out_subdirs = [out_dir / subdir.name for subdir in in_subdirs]

        # get all hazen tasks selected by user
        tasks_to_run = self.checkbutton_panel_tasks.get_selected_items()

        # get baselines from table
        baselines = self.table_baselines.get_as_pandas_df()

        return in_dir, out_dir, in_subdirs, out_subdirs, tasks_to_run, baselines

    @staticmethod
    def _config_settings_valid(
        in_dir, out_dir, in_subdirs, out_subdirs, tasks_to_run, baselines
    ):
        # Check that in_dir is not "." (the fallback option from gui) and that it also exists and is a directory
        if str(in_dir) == "." or not in_dir.is_dir() or not in_dir.exists():
            tk.messagebox.showerror("Error", "Input directory is invalid!")
            return False

        # Check that out_dir is not "." (the fallback option from gui) and that it's a directory
        if str(out_dir) == "." or not out_dir.is_dir():
            tk.messagebox.showerror("Error", "Output directory is invalid!")
            return False

        # check that 1+ subdirectories have been selected
        if not in_subdirs:
            tk.messagebox.showerror("Error", "No subdirectories selected!")
            return False

        # if no tasks selected, cancel process
        if not tasks_to_run:
            tk.messagebox.showerror("Error", "No tasks selected!")
            return False

        # check to see if any baseline values are missing
        if (baselines == "").any().any():
            tk.messagebox.showerror("Error", "Values missing in baseline table!")
            return False

        # check file structure before running
        if file_structure_problems_exist(in_subdirs, tasks_to_run):
            return False

        # if out_dir doesn't exist, ask user whether they want to proceed
        elif not out_dir.exists():
            proceed = tk.messagebox.askyesno(
                "Confirm",
                "Output directory does not currently exist!\nCreate directory and run?",
            )

            # if wants to proceed, make out_dir
            if proceed:
                out_dir.mkdir()

            # otherwise cancel process and go back to responsive gui
            else:
                return False

        # Check whether any output subdirectories do not exist, and if the user is happy to overwrite contents
        if any(out_subdir.exists() for out_subdir in out_subdirs):
            overwrite = tk.messagebox.askyesno(
                "Confirm",
                "Some output subdirectories already exist!\nOverwrite where necessary?",
            )

            # if user does not want to overwrite, cancel process
            if not overwrite:
                return False

        # otherwise make all output subdirectories
        for out_subdir in out_subdirs:
            out_subdir.mkdir(exist_ok=True)

        return True

    def handle_event(self, event: tuple):
        """Handles specific events passed through from main event
        queue (see App._check_queue).

        Args:
            event (tuple): Tuple containing unique event trigger strings
                (see shared/queueing.py).
        """

        # Handle events relating to a progress bar update
        if event[0] == "PROGRESS_BAR_UPDATE":

            # Triggers progress bar update during DICOM checking and sorting tasks
            if event[1] in ("DICOM_CHECKING", "DICOM_SORTING"):
                self.modal_progress.add_progress(event[2])

            # If try to access a progress bar that is not recognised, throw error
            else:
                raise ValueError(f"Invalid progress bar ID: {event[1]}")

        # Handle events relating to a particular task being completed
        if event[0] == "TASK_COMPLETE":

            # Triggers when software finished checking for valid DICOMs
            if event[1] == "DICOM_CHECKING":

                # destroys modal progress bar window for dicom checking and creates a new one for dicom sorting
                self.modal_progress.destroy()
                self.modal_progress = ModalProgressBar(self, "Sorting loose DICOMs")

            # Triggers when software finished sorting loose DICOMs
            elif event[1] == "DICOM_SORTING":

                # destroys modal progress bar window for dicom sorting
                self.modal_progress.destroy()

                # populates listbox with subdirectories of input directory
                in_subdirs = [p.name for p in self.in_dir.glob("*") if p.is_dir()]
                self.scrollable_listbox_subdirs.populate(in_subdirs)
