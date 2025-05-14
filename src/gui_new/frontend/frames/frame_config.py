from pathlib import Path

import tkinter as tk
from tkinter import filedialog
import threading

from shared.context import AVAILABLE_TASKS
from shared.queueing import get_queue
from frontend.settings import FONT_TEXT, FONT_TITLE
from frontend.progress_bar_modal import ProgressBarModal
from backend.sort_dicoms import sort_dicoms


class FrameConfig(tk.Frame):
    """Subclass of tk.Frame.
    For the user to configure settings for AutoQC_ACR application.

    Class attributes:
        GRID_DIMS (tuple): A tuple of expected grid dimensions for tk.widget.grid()
        PAD_X (tuple): Standardised x-padding between grid columns.
        PAD_Y (tuple): Standardised y-padding between grid rows.
    """

    GRID_DIMS = (6, 3)
    PAD_X = (0, 10)
    PAD_Y = (0, 15)

    def __init__(self, master):
        super().__init__(master)

        self._create_widgets()
        self._configure_widgets()
        self._layout_widgets()
        self._configure_grid()
        self.modal_progress = None

    def _create_widgets(self):
        """Lays out widgets within self."""
        # Frames
        self.frame_task_checkboxes = tk.Frame(self, bd=1, relief=tk.SOLID)

        # Labels
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

        # Entries
        self.entry_in_dir = tk.Entry(self, font=FONT_TEXT)
        self.entry_out_dir = tk.Entry(self, font=FONT_TEXT)

        # Listboxes
        self.listbox_subdirs = tk.Listbox(
            self,
            selectmode=tk.EXTENDED,
            font=FONT_TEXT,
            selectforeground="black",
            selectbackground="lightgray",
            bd=1,
            relief=tk.SOLID,
        )

        # Scrollbars
        self.scrollbar_subdirs = tk.Scrollbar(
            self,
            orient=tk.VERTICAL,
            width=20,
            bd=2,
            relief=tk.SOLID,
            command=self.listbox_subdirs.yview,
        )

        # Checkboxes
        self.task_checkboxes_vars = [tk.BooleanVar(value=True) for _ in AVAILABLE_TASKS]
        self.task_checkboxes = [
            tk.Checkbutton(
                self.frame_task_checkboxes,
                text=task_name,
                font=FONT_TEXT,
                variable=var,
            )
            for var, task_name in zip(self.task_checkboxes_vars, AVAILABLE_TASKS)
        ]

        # Buttons
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
            command=self._read_config_settings,
            bd=1,
            relief=tk.SOLID,
        )

    def _configure_widgets(self):
        """Configures widgets."""
        self.listbox_subdirs.config(yscrollcommand=self.scrollbar_subdirs.set)

    def _layout_widgets(self):
        """Lays out widgets in self."""
        # Column 0
        self.label_title.grid(
            row=0,
            column=0,
            columnspan=self.GRID_DIMS[0],
            sticky="w",
            pady=self.PAD_Y,
        )
        self.label_in_dir.grid(
            row=1, column=0, sticky="w", padx=self.PAD_X, pady=self.PAD_Y
        )
        self.label_out_dir.grid(
            row=2, column=0, sticky="w", padx=self.PAD_X, pady=self.PAD_Y
        )
        self.label_select_subdirs.grid(
            row=3, column=0, sticky="nw", padx=self.PAD_X, pady=self.PAD_Y
        )
        self.label_select_tasks.grid(
            row=4, column=0, sticky="nw", padx=self.PAD_X, pady=self.PAD_Y
        )
        self.button_run.grid(
            row=5, column=0, columnspan=self.GRID_DIMS[0], sticky="ew", padx=50
        )

        # Column 1
        self.entry_in_dir.grid(
            row=1, column=1, sticky="ew", padx=self.PAD_X, pady=self.PAD_Y
        )
        self.entry_out_dir.grid(
            row=2, column=1, sticky="ew", padx=self.PAD_X, pady=self.PAD_Y
        )
        self.listbox_subdirs.grid(
            row=3, column=1, sticky="nsew", padx=self.PAD_X, pady=self.PAD_Y
        )
        self.frame_task_checkboxes.grid(
            row=4, column=1, sticky="nsw", padx=self.PAD_X, pady=self.PAD_Y
        )

        # Column 2
        self.button_browse_in_dir.grid(row=1, column=2, sticky="w", pady=self.PAD_Y)
        self.button_browse_out_dir.grid(row=2, column=2, sticky="w", pady=self.PAD_Y)
        self.scrollbar_subdirs.grid(row=3, column=2, sticky="nsw", pady=self.PAD_Y)

        # Other layout
        for task_checkbox in self.task_checkboxes:
            task_checkbox.pack(anchor="w", padx=self.PAD_Y)

    def _configure_grid(self):
        """Configures grid weights and min row/col sizes (within self)."""
        self.columnconfigure(0, weight=0, minsize=0)
        self.columnconfigure(1, weight=1, minsize=400)
        self.columnconfigure(2, weight=0, minsize=0)

        self.rowconfigure(0, weight=0, minsize=0)
        self.rowconfigure(1, weight=0, minsize=0)
        self.rowconfigure(2, weight=0, minsize=50)
        self.rowconfigure(3, weight=1, minsize=0)

    def _populate_entry_widget(self, entry_widget: tk.Entry, text: str):
        """Populates an entry widget with a string, deleting current content.

        Args:
            entry_widget (tk.Entry): Entry widget to populate.
            text (str): String to populate entry widget with.
        """
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, text)
        entry_widget.icursor(tk.END)
        entry_widget.xview_moveto(1)

    def _populate_listbox_with_subdirs(self, listbox: tk.Listbox, in_dir: Path):
        """Populates a listbox widget with the subdirectories of a particular parent dir.

        Args:
            listbox (tk.Listbox): Listbox to populate with subdirectories.
            in_dir (Path): Path to parent directory.
        """
        listbox.delete(0, tk.END)
        for subdir in in_dir.iterdir():
            if subdir.is_dir():
                listbox.insert(tk.END, subdir.name)

    def _browse_in_dir(self):
        """Asks user to choose an input directory. This should be a directory full of
        sorted or unsorted DICOMs. Populates self.entry_in_dir with the users choice,
        populates self.entry_out_dir with default output dir and starts a thread to
        execute DICOM sorting process.
        """
        self.in_dir = Path(filedialog.askdirectory(title="Select Input Directory"))
        if self.in_dir:
            self._populate_entry_widget(self.entry_in_dir, str(self.in_dir.resolve()))
            self._populate_entry_widget(
                self.entry_out_dir,
                str((self.in_dir.parent / "AutoQC_ACR_Output").resolve()),
            )
            self.modal_progress = ProgressBarModal(self, "Checking for DICOMs")
            threading.Thread(target=sort_dicoms, args=(self.in_dir,)).start()

    def _browse_out_dir(self):
        """Asks user to choose an output directory for results.
        Populates self.entry_out_dir with the users choice
        """
        out_dir = Path(filedialog.askdirectory(title="Select Output Directory"))
        if out_dir:
            self._populate_entry_widget(self.entry_out_dir, str(out_dir.resolve()))

    def _read_config_settings(self):
        """Pulls configuration settings from widgets after performing validation checks
        on relevant widgets. Sends a message to global queue to switch frame to
        'TASKRUNNER' frame, also passing configuration settings.
        """
        in_dir = Path(self.entry_in_dir.get())
        if str(in_dir) == "." or not in_dir.is_dir() or not in_dir.exists():
            tk.messagebox.showerror("Error", "Input directory is invalid!")
            return

        # Check for valid in_subdir selection
        in_subdirs = [
            in_dir / self.listbox_subdirs.get(i)
            for i in self.listbox_subdirs.curselection()
        ]
        if not in_subdirs:
            tk.messagebox.showerror("Error", "No subdirectories selected!")
            return

        # Check for valid out_dir selection
        out_dir = Path(self.entry_out_dir.get())
        if str(out_dir) == "." or not in_dir.is_dir():
            tk.messagebox.showerror("Error", "Output directory is invalid!")
            return
        elif not out_dir.exists():
            proceed = tk.messagebox.askyesno(
                "Confirm",
                "Output directory does not currently exist!\nCreate directory and run?",
            )
            if proceed:
                out_dir.mkdir()
            else:
                return

        # Make out_subdirs if don't exist
        out_subdirs = [out_dir / subdir.name for subdir in in_subdirs]

        if any(out_subdir.exists() for out_subdir in out_subdirs):
            overwrite = tk.messagebox.askyesno(
                "Confirm",
                "Some output subdirectories already exist!\nOverwrite where necessary?",
            )
            if not overwrite:
                return

        for out_subdir in out_subdirs:
            out_subdir.mkdir(exist_ok=True)

        # Check that at least one task is selected
        tasks_to_run = [
            task_checkbox.cget("text")
            for task_checkbox, var in zip(
                self.task_checkboxes, self.task_checkboxes_vars
            )
            if var.get() == 1
        ]
        if not tasks_to_run:
            tk.messagebox.showerror("Error", "No tasks selected!")
            return

        args_to_pass = (in_dir, out_dir, in_subdirs, out_subdirs, tasks_to_run)
        get_queue().put(("SWITCH_FRAME", "TASKRUNNER", args_to_pass))

    def handle_event(self, event: tuple):
        """Handles events passed from main event queue (see App._check_queue).

        Args:
            event (tuple): Tuple containing unique event trigger strings.
        """
        if event[0] == "PROGRESS_BAR_UPDATE":
            # Triggers progress bar update during DICOM checking and sorting tasks.
            if event[1] in ("DICOM_CHECKING", "DICOM_SORTING"):
                self.modal_progress.add_progress(event[2])
            else:
                raise ValueError(f"Invalid progress bar ID: {event[1]}")

        if event[0] == "TASK_COMPLETE":
            # Triggers task completion events for DICOM checking and sorting tasks.
            if event[1] == "DICOM_CHECKING":
                # destroys current progress bar window and creates another for DICOM sorting task.
                self.modal_progress.destroy()
                self.modal_progress = ProgressBarModal(self, "Sorting loose DICOMs")
            elif event[1] == "DICOM_SORTING":
                # destroys current progress bar window and populates listbox with subdirectories of input dir.
                self.modal_progress.destroy()
                self._populate_listbox_with_subdirs(self.listbox_subdirs, self.in_dir)
