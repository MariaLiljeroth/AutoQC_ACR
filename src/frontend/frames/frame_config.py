"""
frame_config.py

This file defines the FrameConfig tk.Frame subclass, whose purpose is to allow the user to configure AutoQC_ACR settings before running it.
This frame is the first frame to appear within the App instance, and is swapped out to FrameTaskRunner instance once hazen tasks begin running.

Written by Nathan Crossley 2025

"""

from pathlib import Path
import threading

import tkinter as tk
from tkinter import filedialog

from shared.context import AVAILABLE_TASKS
from shared.queueing import get_queue

from frontend.settings import FONT_TEXT, FONT_TITLE
from frontend.progress_bar_modal import ProgressBarModal

from backend.dcm_sorter import DcmSorter


class FrameConfig(tk.Frame):
    """Subclass of tk.Frame that allows the user to configure settings for
    AutoQC_ACR, before the tasks begin running. The user sets the input dicom directory,
    output dicom directory and selects the sorted subdirectories to analyse. The
    specific tasks to run are also selected.

    Instance attributes:
        frames (dict[str, tk.Frame]): Dictionary of frame widgets.
        labels (dict[str, tk.Label]): Dictionary of label widgets.
        entries (dict[str, tk.Entry]): Dictionary of entry widgets.
        listboxes (dict[str, tk.Listbox]): Dictionary of listbox widgets.
        scrollbars (dict[str, tk.Scrollbar]): Dictionary of scrollbar widgets.
        checkbuttons (dict[str, dict[str, list[tk.Checkbutton | tk.BooleanVar]]]): Dictionary of checkbutton widgets and associated boolean vars.
        buttons (dict[str, tk.Button]): Dictionary of button widgets.
        modal_progress (ProgressBarModal): Modal progress bar window for displaying progress during DICOM checking and sorting tasks.
        in_dir (Path): Input directory selected by the user (should contain unsorted DICOMs or folders previously sorted).

    Class attributes:
        GRID_DIMS (tuple): A tuple of expected grid dimensions for grid within self.
        PAD_X (tuple): Standardised x-padding between grid columns.
        PAD_Y (tuple): Standardised y-padding between grid rows.
    """

    GRID_DIMS = (6, 3)
    PAD_X = (0, 10)
    PAD_Y = (0, 15)

    def __init__(self, master: tk.Tk):
        """Initialises FrameConfig instance. Creates, configures
        and lays out widgets within self.

        Args:
            master (tk.Tk): Root window from on which self is placed.
        """
        super().__init__(master)

        # Create, configure and layout widgets within self.
        self._create_widgets()
        self._configure_widgets()
        self._layout_widgets()

        # Configure grid dimensions within self.
        self._configure_grid()

        # Set to None as no modal progress bar exists at initialisation.
        self.modal_progress = None

    def _create_widgets(self):
        """Creates widgets within self."""

        # Dict containing string-id and widget pairs for all frames within self.
        self.frames = {"task_checkbuttons": tk.Frame(self, bd=1, relief=tk.SOLID)}

        # Dict containing string-id and widget pairs for all labels within self.
        # Labels are used to convey information about how the user should interact with the GUI.
        self.labels = {
            "title": tk.Label(
                self,
                text="Configuration Settings",
                font=FONT_TITLE,
                anchor="w",
            ),
            "in_dir": tk.Label(self, text="Input directory:", font=FONT_TEXT),
            "out_dir": tk.Label(self, text="Output directory:", font=FONT_TEXT),
            "select_subdirs": tk.Label(
                self,
                text="Select subdirectories\nto process:",
                font=FONT_TEXT,
                justify=tk.LEFT,
            ),
            "select_tasks": tk.Label(self, text="Select tasks to run:", font=FONT_TEXT),
        }

        # Dict containing string-id and widget pairs for all entries within self.
        # Seperate entries exist for displaying the selected input and output toplevel directories.
        self.entries = {
            "in_dir": tk.Entry(self, font=FONT_TEXT),
            "out_dir": tk.Entry(self, font=FONT_TEXT),
        }

        # Dict containing string-id and widget pairs for all listboxes within self.
        # subdirs listbox used to select which sorted DICOM subdirectories are used for task running.
        self.listboxes = {
            "subdirs": tk.Listbox(
                self,
                selectmode=tk.EXTENDED,
                font=FONT_TEXT,
                selectforeground="black",
                selectbackground="lightgray",
                bd=1,
                relief=tk.SOLID,
            )
        }

        # Dict containing string-id and widget pairs for all scrollbars within self.
        # subdirs scrollbar scrolls the subdirs listbox for the case of many dirs displayed.
        self.scrollbars = {
            "subdirs": tk.Scrollbar(
                self,
                orient=tk.VERTICAL,
                width=20,
                bd=2,
                relief=tk.SOLID,
                command=self.listboxes["subdirs"].yview,
            )
        }

        # Create a boolean var associated with every task to track state of each task tickbox.
        bool_vars_tasks = [tk.BooleanVar(value=True) for _ in AVAILABLE_TASKS]

        # Dict containing info for all checkbuttons within self.
        # The toplevel key represents the string-id for a given set of checkbuttons.
        # The lower "checkbuttons" key returns the list of checkbox widgets and "bool_vars" returns their state.
        self.checkbuttons = {
            "tasks": {
                "checkbuttons": [
                    tk.Checkbutton(
                        self.frames["task_checkbuttons"],
                        text=task_name,
                        font=FONT_TEXT,
                        variable=var,
                    )
                    for var, task_name in zip(bool_vars_tasks, AVAILABLE_TASKS)
                ],
                "bool_vars": bool_vars_tasks,
            },
        }

        # Dict containing string-id and widget pairs for all buttons within self.
        # buttons exist for browsing input and output directories plus for initialising task running process.
        self.buttons = {
            "browse_in_dir": tk.Button(
                self,
                text="Browse",
                font=FONT_TEXT,
                command=self._browse_in_dir,
                bd=1,
                relief=tk.SOLID,
            ),
            "browse_out_dir": tk.Button(
                self,
                text="Browse",
                font=FONT_TEXT,
                command=self._browse_out_dir,
                bd=1,
                relief=tk.SOLID,
            ),
            "run": tk.Button(
                self,
                text="Run AutoQC_ACR",
                font=FONT_TEXT,
                command=self._read_config_settings,
                bd=1,
                relief=tk.SOLID,
            ),
        }

    def _configure_widgets(self):
        """Configures widgets within self"""
        # link subdirs listbox to subdirs scrollbar.
        self.listboxes["subdirs"].config(yscrollcommand=self.scrollbars["subdirs"].set)

    def _layout_widgets(self):
        """Lays out widgets within self."""
        # position all labels within grid of self
        for key, widget in self.labels.items():
            if key == "title":
                widget.grid(
                    row=0,
                    column=0,
                    columnspan=self.GRID_DIMS[0],
                    sticky="w",
                    pady=self.PAD_Y,
                )
            elif key == "in_dir":
                widget.grid(
                    row=1, column=0, sticky="w", padx=self.PAD_X, pady=self.PAD_Y
                )
            elif key == "out_dir":
                widget.grid(
                    row=2, column=0, sticky="w", padx=self.PAD_X, pady=self.PAD_Y
                )
            elif key == "select_subdirs":
                widget.grid(
                    row=3, column=0, sticky="nw", padx=self.PAD_X, pady=self.PAD_Y
                )
            elif key == "select_tasks":
                widget.grid(
                    row=4, column=0, sticky="nw", padx=self.PAD_X, pady=self.PAD_Y
                )

        # position all buttons within grid of self
        for key, widget in self.buttons.items():
            if key == "browse_in_dir":
                widget.grid(row=1, column=2, sticky="w", pady=self.PAD_Y)
            elif key == "browse_out_dir":
                widget.grid(row=2, column=2, sticky="w", pady=self.PAD_Y)
            elif key == "run":
                widget.grid(
                    row=5, column=0, columnspan=self.GRID_DIMS[0], sticky="ew", padx=50
                )

        # position all frames within grid of self
        for key, widget in self.frames.items():
            if key == "task_checkbuttons":
                widget.grid(
                    row=4, column=1, sticky="nsw", padx=self.PAD_X, pady=self.PAD_Y
                )

        # position all entries within grid of self
        for key, widget in self.entries.items():
            if key == "in_dir":
                widget.grid(
                    row=1, column=1, sticky="ew", padx=self.PAD_X, pady=self.PAD_Y
                )
            elif key == "out_dir":
                widget.grid(
                    row=2, column=1, sticky="ew", padx=self.PAD_X, pady=self.PAD_Y
                )

        # position all listboxes within grid of self
        for key, widget in self.listboxes.items():
            if key == "subdirs":
                widget.grid(
                    row=3, column=1, sticky="nsew", padx=self.PAD_X, pady=self.PAD_Y
                )

        # position all scrollbars within grid of self
        for key, widget in self.scrollbars.items():
            if key == "subdirs":
                widget.grid(row=3, column=2, sticky="nsw", pady=self.PAD_Y)

        # position all lcheckboxes within their frame.
        for key, widget in self.checkbuttons.items():
            if key == "tasks":
                for checkbutton in widget["checkbuttons"]:
                    checkbutton.pack(anchor="w", padx=self.PAD_Y)

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

    def _populate_entry_widget(self, entry_widget: tk.Entry, text: str):
        """Populates an entry widget with a text string, deleting current content.

        Args:
            entry_widget (tk.Entry): Entry widget to populate.
            text (str): String to populate entry widget with.
        """
        # delete all content in entry widget
        entry_widget.delete(0, tk.END)

        # insert text within entry widget
        entry_widget.insert(0, text)

        # ensure that start of entry widget's text is visible
        entry_widget.icursor(tk.END)
        entry_widget.xview_moveto(1)

    def _populate_listbox_with_subdirs(self, listbox: tk.Listbox, dir: Path):
        """Populates a listbox widget with the subdirectories of a particular parent dir.

        Args:
            listbox (tk.Listbox): Listbox to populate with subdirectories.
            dir (Path): Path to parent directory.
        """

        # remove all current listbox items
        listbox.delete(0, tk.END)

        # get all subdirs or parent dir and append to listbox
        for subdir in dir.iterdir():
            if subdir.is_dir():
                listbox.insert(tk.END, subdir.name)

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
            self._populate_entry_widget(
                self.entries["in_dir"], str(self.in_dir.resolve())
            )

            # populates out_dir entry widget with default output directory based on input directory
            self._populate_entry_widget(
                self.entries["out_dir"],
                str((self.in_dir.parent / "AutoQC_ACR_Output").resolve()),
            )

            # Instantiates a modal window for progress bar
            self.modal_progress = ProgressBarModal(self, "Checking for DICOMs")

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
            self._populate_entry_widget(self.entries["out_dir"], str(out_dir.resolve()))

    def _read_config_settings(self):
        """Pulls configuration settings from widgets after performing relevant
        validation checks. Sends a message to global queue to switch frame to
        'TASKRUNNER', also passing configuration settings.
        """

        # get input directory from relevant entry widget
        in_dir = Path(self.entries["in_dir"].get())

        # Check that in_dir is not "." (the fallback option from gui) and that it also exists and is a directory
        if str(in_dir) == "." or not in_dir.is_dir() or not in_dir.exists():
            tk.messagebox.showerror("Error", "Input directory is invalid!")
            return

        # get input subdirectories from relevant listbox
        in_subdirs = [
            in_dir / self.listboxes["subdirs"].get(i)
            for i in self.listboxes["subdirs"].curselection()
        ]

        # check that 1+ subdirectories have been selected
        if not in_subdirs:
            tk.messagebox.showerror("Error", "No subdirectories selected!")
            return

        # get output directory from relevant entry widget
        out_dir = Path(self.entries["out_dir"].get())

        # Check that out_dir is not "." (the fallback option from gui) and that it's a directory
        if str(out_dir) == "." or not in_dir.is_dir():
            tk.messagebox.showerror("Error", "Output directory is invalid!")
            return

        # if doesn't exist, ask user whether they want to proceed
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
                return

        # infer output subdirectories from input subdirectories
        out_subdirs = [out_dir / subdir.name for subdir in in_subdirs]

        # Check whether any output subdirectories do not exist, and if the user is happy to overwrite contents
        if any(out_subdir.exists() for out_subdir in out_subdirs):
            overwrite = tk.messagebox.askyesno(
                "Confirm",
                "Some output subdirectories already exist!\nOverwrite where necessary?",
            )

            # if user does not want to overwrite, cancel process
            if not overwrite:
                return

        # make all output subdirectories
        for out_subdir in out_subdirs:
            out_subdir.mkdir(exist_ok=True)

        # get all hazen tasks selected by user
        tasks_to_run = [
            check_button.cget("text")
            for check_button, var in zip(
                self.checkbuttons["tasks"]["checkbuttons"],
                self.checkbuttons["tasks"]["bool_vars"],
            )
            if var.get() == 1
        ]

        # if no tasks selected, cancel process
        if not tasks_to_run:
            tk.messagebox.showerror("Error", "No tasks selected!")
            return

        # construct tuple of args to pass to taskrunner frame
        args_to_pass = (in_dir, out_dir, in_subdirs, out_subdirs, tasks_to_run)

        # signal a switch to taskrunner frame and pass args
        get_queue().put(("SWITCH_FRAME", "TASKRUNNER", args_to_pass))

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
                self.modal_progress = ProgressBarModal(self, "Sorting loose DICOMs")

            # Triggers when software finished sorting loose DICOMs
            elif event[1] == "DICOM_SORTING":

                # destroys modal progress bar window for dicom sorting
                self.modal_progress.destroy()

                # populates listbox with subdirectories of input directory
                self._populate_listbox_with_subdirs(
                    self.listboxes["subdirs"], self.in_dir
                )
