"""
page_config.py

This file defines the PageConfig tk.Frame subclass, whose purpose is to allow the user to configure AutoQC_ACR settings before running it.
This page is the first page to appear within the App instance, and is swapped out to PageJobRunner instance once hazen jobs begin running.

Written by Nathan Crossley 2025

"""

from pathlib import Path
import threading

import tkinter as tk
from tkinter import filedialog

from src.shared.context import AVAILABLE_TASKS, EXPECTED_COILS
from src.shared.queueing import get_queue, QueueTrigger

from src.frontend.app_state import AppState
from src.frontend.settings import FONT_TEXT, FONT_TITLE, PAD_MEDIUM, PAD_LARGE
from src.frontend.widgets.modal_progress_bar import ModalProgressBar
from src.frontend.widgets.scrollable_listbox import ScrollableListbox
from src.frontend.widgets.checkbutton_panel import CheckbuttonPanel
from src.frontend.widgets.simple_table import SimpleTable
from src.frontend.tk_utils import populate_entry_widget

from src.backend.dcm_sorter import DcmSorter
from src.backend.configuration_tests import file_structure_problems_exist


class PageConfig(tk.Frame):
    """tk.Frame subclass representing page for AutoQC_ACR settings
    configuration.
    """

    def __init__(self, master: tk.Tk, app_state: AppState):
        """Initialises PageConfig instance. Creates, configures
        and lays out widgets within self.

        Args:
            master (tk.Tk): Root window within which self is placed.
            app_state (AppState): class for sharing app state within frontend
        """
        super().__init__(master)

        # store reference to passed app state class.
        self.app_state = app_state

        # Create, configure and layout widgets within self.
        self._create_widgets()
        self._layout_widgets()

        # Configure grid dimensions within self.
        self._configure_grid()

        # Set to None as no modal progress bar exists at initialisation.
        self.modal_progress = None

    def _create_widgets(self):
        """Creates widgets within self. Individual functions
        are called for different types of widget."""
        self._create_labels()
        self._create_entries()
        self._create_scrollable_listboxes()
        self._create_checkbutton_panels()
        self._create_buttons()
        self._create_tables()

    def _create_labels(self):
        """Creates all labels required for this page."""

        # create label for the page title
        self.label_title = tk.Label(
            self,
            text="Configuration Settings",
            font=FONT_TITLE,
            anchor="w",
        )

        # create label telling user to browse for an input directory
        self.label_in_dir = tk.Label(self, text="Input directory:", font=FONT_TEXT)

        # create label telling user to browse for an output directory
        self.label_out_dir = tk.Label(self, text="Output directory:", font=FONT_TEXT)

        # create label telling user to select input subdirectories
        self.label_select_subdirs = tk.Label(
            self,
            text="Select subdirectories\nto process:",
            font=FONT_TEXT,
            justify=tk.LEFT,
        )

        # create label telling user to select tasks
        self.label_select_tasks = tk.Label(
            self, text="Select tasks to run:", font=FONT_TEXT
        )

        # create label telling user to input baselines
        self.label_input_baselines = tk.Label(
            self, text="Input baselines:", font=FONT_TEXT
        )

    def _create_entries(self):
        """Creates all entries required for this page."""

        # entry for storing the user's input directory selection
        self.entry_in_dir = tk.Entry(self, font=FONT_TEXT, bd=1, relief=tk.SOLID)

        # entry for storing the user's output directory selection
        self.entry_out_dir = tk.Entry(self, font=FONT_TEXT, bd=1, relief=tk.SOLID)

    def _create_scrollable_listboxes(self):
        """Creates all scrollable listboxes required for this page."""

        # scrollable listbox for user to select input subdirectories.
        self.scrollable_listbox_subdirs = ScrollableListbox(self)

    def _create_checkbutton_panels(self):
        """Creates all checkbutton panels required for this page."""

        # checkbutton panel for user to select tasks that they want to run
        self.checkbutton_panel_tasks = CheckbuttonPanel(
            self, checkbutton_names=AVAILABLE_TASKS
        )

    def _create_buttons(self):
        """Creates all buttons required for this page."""

        # button to allow user to browse for an input directory
        self.button_browse_in_dir = tk.Button(
            self,
            text="Browse",
            font=FONT_TEXT,
            command=self._browse_in_dir,
            bd=1,
            relief=tk.SOLID,
        )

        # button to allow user to browse for an output directory
        self.button_browse_out_dir = tk.Button(
            self,
            text="Browse",
            font=FONT_TEXT,
            command=self._browse_out_dir,
            bd=1,
            relief=tk.SOLID,
        )

        # button to allow user to run AutoQC_ACR (end configuration stage)
        self.button_run = tk.Button(
            self,
            text="Run AutoQC_ACR",
            font=FONT_TEXT,
            command=self._trigger_job_running,
            bd=1,
            relief=tk.SOLID,
        )

    def _create_tables(self):
        """Create all tables required for this page."""

        # table to allow user to enter baseline values.
        uniformity_header = AVAILABLE_TASKS[3]
        snr_header = AVAILABLE_TASKS[1]

        self.table_baselines = SimpleTable(
            self,
            [snr_header, uniformity_header],
            EXPECTED_COILS,
            border_pad=PAD_LARGE,
            bd=1,
            relief=tk.SOLID,
        )

    def _layout_widgets(self):
        """Layout all widgets within self. Individual
        functions are called for individual types of widget."""

        self._layout_labels()
        self._layout_entries()
        self._layout_scrollable_listboxes()
        self._layout_checkbutton_panels()
        self._layout_buttons()
        self._layout_tables()

    def _layout_labels(self):
        """Layout all labels for this page"""
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
        """Layout all entries for this page."""
        self.entry_in_dir.grid(
            row=1, column=1, sticky="ew", padx=(0, PAD_MEDIUM), pady=(0, PAD_LARGE)
        )
        self.entry_out_dir.grid(
            row=2, column=1, sticky="ew", padx=(0, PAD_MEDIUM), pady=(0, PAD_LARGE)
        )

    def _layout_scrollable_listboxes(self):
        """Layout all scrollable listboxes for this page."""
        self.scrollable_listbox_subdirs.grid(
            row=3, column=1, sticky="ew", padx=(0, PAD_MEDIUM), pady=(0, PAD_LARGE)
        )

    def _layout_checkbutton_panels(self):
        """Layout all checkbutton panels for this page."""
        self.checkbutton_panel_tasks.grid(
            row=4, column=1, sticky="w", padx=(0, PAD_MEDIUM), pady=(0, PAD_LARGE)
        )

    def _layout_buttons(self):
        """Layout all buttons for this page."""
        self.button_browse_in_dir.grid(row=1, column=2, sticky="w", pady=(0, PAD_LARGE))
        self.button_browse_out_dir.grid(
            row=2, column=2, sticky="w", pady=(0, PAD_LARGE)
        )
        self.button_run.grid(row=6, column=0, columnspan=3, sticky="ew", padx=50)

    def _layout_tables(self):
        """Layout all tables for this page."""
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

        # get user to select input directory through basic gui window
        current_in_dir = Path(filedialog.askdirectory(title="Select Input Directory"))

        if current_in_dir:
            # populates in_dir entry widget with selected input directory
            populate_entry_widget(self.entry_in_dir, str(current_in_dir.resolve()))

            # populates out_dir entry widget with default output directory based on input directory
            populate_entry_widget(
                self.entry_out_dir,
                str((current_in_dir.parent / "AutoQC_ACR_Output").resolve()),
            )

            # Instantiates a modal window for progress bar
            self.modal_progress = ModalProgressBar(self, "Checking for DICOMs")

            # Instantiates an instance of class for sorting DICOMs
            ds = DcmSorter(current_in_dir)

            # save as instance attribute for later
            self.current_in_dir = current_in_dir

            # starts running DICOM sorting process in separate thread to prevent blocking gui
            threading.Thread(target=ds.run).start()

    def _browse_out_dir(self):
        """Asks user to choose an output directory for results (activated through button).
        Populates self.entry_out_dir with the user's choice.
        """

        # gets user to select output directory through gui
        out_dir = Path(filedialog.askdirectory(title="Select Output Directory"))

        if out_dir:
            # populate output directory entry widget with user's choice
            populate_entry_widget(self.entry_out_dir, str(out_dir.resolve()))

    def _trigger_job_running(self):
        """Reads configuration settings from this page and stores to app state class.
        Checks to see if any configuration settings are invalid and if not triggers
        job running pipeline.
        """

        # read configuration settings from this page and store to app state
        self._read_config_settings()

        # check whether all configuration settings are valid or not.
        valid = self._config_settings_valid()

        # if all settings are valid, send a trigger to the queue, asking to
        # switch pages to taskrunner page.
        if valid:
            get_queue().put(QueueTrigger("SWITCH_PAGE", "TASKRUNNER"))

        # else go back to responsive configuration page for ammendment.
        else:
            return

    def _read_config_settings(self):
        """Pulls configuration settings from widgets on this page and
        stores them in app state for use in other pages.
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
        baselines = self.table_baselines.get_current_state()

        # store all configuration settings in app state, so can be used within any page.
        self.app_state.in_dir = in_dir
        self.app_state.out_dir = out_dir
        self.app_state.in_subdirs = in_subdirs
        self.app_state.out_subdirs = out_subdirs
        self.app_state.tasks_to_run = tasks_to_run
        self.app_state.baselines = baselines

    def _config_settings_valid(self) -> bool:
        """Checks whether all configuration settings in app state
        class are valid to initiate job running pipeline.

        Returns:
            bool: True if all configuration settings are valid, False otherwise.
        """

        # get local copies of all app state attributes needed for validation checks.
        in_dir = self.app_state.in_dir
        out_dir = self.app_state.out_dir
        in_subdirs = self.app_state.in_subdirs
        out_subdirs = self.app_state.out_subdirs
        tasks_to_run = self.app_state.tasks_to_run
        baselines = self.app_state.baselines

        # quit if in_dir is "." (the fallback option from gui) or if its not a directory.
        if str(in_dir) == "." or not in_dir.is_dir() or not in_dir.exists():
            tk.messagebox.showerror("Error", "Input directory is invalid!")
            return False

        # quit if out_dir is "." (the fallback option from gui) or if its not a directory.
        if str(out_dir) == "." or not out_dir.is_dir():
            tk.messagebox.showerror("Error", "Output directory is invalid!")
            return False

        # quit if no subdirectories selected.
        if not in_subdirs:
            tk.messagebox.showerror("Error", "No subdirectories selected!")
            return False

        # quit if no tasks selected
        if not tasks_to_run:
            tk.messagebox.showerror("Error", "No tasks selected!")
            return False

        # quit if any baselines are missing
        if (baselines == "").any().any():
            tk.messagebox.showerror("Error", "Values missing in baseline table!")
            return False

        # quit if any problems with file structure.
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

        # if all checks pass, return True
        return True

    def handle_task_request(self, trigger: QueueTrigger):
        """Handles specific page-specific triggers passed through
        from main queue (see App._check_queue).

        Args:
            trigger (QueueTrigger): queue trigger to activate.
        """

        # Handle triggers requesting updating progress bars for backend DICOM sorting and checking
        if (
            trigger.ID == "PROGBAR_UPDATE_DCM_SORTED"
            or trigger.ID == "PROGBAR_UPDATE_DCM_CHECKED"
        ):
            self.modal_progress.add_progress(trigger.data)

        # Handles trigger request to destroy current progress bar and create a new one for DICOM sorting.
        elif trigger.ID == "CREATE_DCM_SORTING_PROGBAR":
            self.modal_progress.destroy()
            self.modal_progress = ModalProgressBar(self, "Sorting loose DICOMs")

        # Handles trigger event to populate scrollable listbox with input subdirectories after DICOM sorting.
        elif trigger.ID == "POPULATE_SUBDIRS_LISTBOX":

            # destroys modal progress bar window for DICOM sorting
            self.modal_progress.destroy()

            # get all names of input subdirectories
            names_in_subdirs = [
                p.name for p in self.current_in_dir.glob("*") if p.is_dir()
            ]

            # populate listbox
            self.scrollable_listbox_subdirs.populate(names_in_subdirs)
