from pathlib import Path

import tkinter as tk
from tkinter import filedialog
import threading

from shared.context import AVAILABLE_TASKS
from shared.queueing import get_queue
from frontend.settings import FONT_TEXT, FONT_TITLE
from frontend.progress_bar_modal import ProgressBarModal
from backend.sort_dicoms import DicomSorter


class FrameConfig(tk.Frame):
    """Subclass of tk.Frame.
    For the user to configure settings for AutoQC_ACR application.

    Instance attributes:
        frames (dict[str, tk.Frame]): Dictionary of frame widgets.
        labels (dict[str, tk.Label]): Dictionary of label widgets.
        entries (dict[str, tk.Entry]): Dictionary of entry widgets.
        listboxes (dict[str, tk.Listbox]): Dictionary of listbox widgets.
        scrollbars (dict[str, tk.Scrollbar]): Dictionary of scrollbar widgets.
        checkbuttons (dict[str, dict[str, list[tk.Checkbutton | tk.BooleanVar]]]): Dictionary of checkbutton widgets and associated boolean vars.
        buttons (dict[str, tk.Button]): Dictionary of button widgets.
        modal_progress (ProgressBarModal): Progress bar modal for displaying progress during DICOM checking and sorting tasks.
        in_dir (Path): Input directory selected by the user.

    Class attributes:
        GRID_DIMS (tuple): A tuple of expected grid dimensions for tk.widget.grid()
        PAD_X (tuple): Standardised x-padding between grid columns.
        PAD_Y (tuple): Standardised y-padding between grid rows.
    """

    GRID_DIMS = (6, 3)
    PAD_X = (0, 10)
    PAD_Y = (0, 15)

    def __init__(self, master: tk.Tk):
        """Initialises FrameConfig instance. Creates, configures and lays out widgets.

        Args:
            master (tk.Tk): Root window of tk application.
        """
        super().__init__(master)

        self._create_widgets()
        self._configure_widgets()
        self._layout_widgets()
        self._configure_grid()
        self.modal_progress = None

    def _create_widgets(self):
        """Creates widgets"""
        self.frames = {"task_checkbuttons": tk.Frame(self, bd=1, relief=tk.SOLID)}
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
        self.entries = {
            "in_dir": tk.Entry(self, font=FONT_TEXT),
            "out_dir": tk.Entry(self, font=FONT_TEXT),
        }
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

        bool_vars_tasks = [tk.BooleanVar(value=True) for _ in AVAILABLE_TASKS]
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
        """Configures widgets."""
        self.listboxes["subdirs"].config(yscrollcommand=self.scrollbars["subdirs"].set)

    def _layout_widgets(self):
        """Lays out widgets."""
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

        for key, widget in self.buttons.items():
            if key == "browse_in_dir":
                widget.grid(row=1, column=2, sticky="w", pady=self.PAD_Y)
            elif key == "browse_out_dir":
                widget.grid(row=2, column=2, sticky="w", pady=self.PAD_Y)
            elif key == "run":
                widget.grid(
                    row=5, column=0, columnspan=self.GRID_DIMS[0], sticky="ew", padx=50
                )

        for key, widget in self.frames.items():
            if key == "task_checkbuttons":
                widget.grid(
                    row=4, column=1, sticky="nsw", padx=self.PAD_X, pady=self.PAD_Y
                )

        for key, widget in self.entries.items():
            if key == "in_dir":
                widget.grid(
                    row=1, column=1, sticky="ew", padx=self.PAD_X, pady=self.PAD_Y
                )
            elif key == "out_dir":
                widget.grid(
                    row=2, column=1, sticky="ew", padx=self.PAD_X, pady=self.PAD_Y
                )

        for key, widget in self.listboxes.items():
            if key == "subdirs":
                widget.grid(
                    row=3, column=1, sticky="nsew", padx=self.PAD_X, pady=self.PAD_Y
                )

        for key, widget in self.scrollbars.items():
            if key == "subdirs":
                widget.grid(row=3, column=2, sticky="nsw", pady=self.PAD_Y)

        for key, widget in self.checkbuttons.items():
            if key == "tasks":
                for checkbutton in widget["checkbuttons"]:
                    checkbutton.pack(anchor="w", padx=self.PAD_Y)

    def _configure_grid(self):
        """Configures grid weights and min row/col sizes for self."""
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
            self._populate_entry_widget(
                self.entries["in_dir"], str(self.in_dir.resolve())
            )
            self._populate_entry_widget(
                self.entries["out_dir"],
                str((self.in_dir.parent / "AutoQC_ACR_Output").resolve()),
            )
            self.modal_progress = ProgressBarModal(self, "Checking for DICOMs")
            ds = DicomSorter(self.in_dir)
            threading.Thread(target=ds.run).start()

    def _browse_out_dir(self):
        """Asks user to choose an output directory for results.
        Populates self.entry_out_dir with the users choice
        """
        out_dir = Path(filedialog.askdirectory(title="Select Output Directory"))
        if out_dir:
            self._populate_entry_widget(self.entries["out_dir"], str(out_dir.resolve()))

    def _read_config_settings(self):
        """Pulls configuration settings from widgets after performing relevant validation checks.
        Sends a message to global queue to switch frame to 'TASKRUNNER', also passing configuration settings.
        """
        in_dir = Path(self.entries["in_dir"].get())
        if str(in_dir) == "." or not in_dir.is_dir() or not in_dir.exists():
            tk.messagebox.showerror("Error", "Input directory is invalid!")
            return

        # Check for valid in_subdir selection
        in_subdirs = [
            in_dir / self.listboxes["subdirs"].get(i)
            for i in self.listboxes["subdirs"].curselection()
        ]
        if not in_subdirs:
            tk.messagebox.showerror("Error", "No subdirectories selected!")
            return

        # Check for valid out_dir selection
        out_dir = Path(self.entries["out_dir"].get())
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
            check_button.cget("text")
            for check_button, var in zip(
                self.checkbuttons["tasks"]["checkbuttons"],
                self.checkbuttons["tasks"]["bool_vars"],
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
                self._populate_listbox_with_subdirs(
                    self.listboxes["subdirs"], self.in_dir
                )
