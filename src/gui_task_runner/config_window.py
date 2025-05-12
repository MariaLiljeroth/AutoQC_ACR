import sys
from pathlib import Path

import pydicom
import tkinter as tk
from tkinter import filedialog

from utils import quick_check_dicom, ModalListProcessor
from context import AVAILABLE_TASKS
from task_runner import TaskRunner


class ConfigWindow(tk.Tk):
    text_font = ("Segoe UI", 11)
    title_font = ("Segoe UI", 14, "bold", "underline")

    def __init__(self):
        super().__init__()
        self._configure_root()
        self._create_widgets()
        self._layout_widgets()
        self._configure_grid()

    def _configure_root(self):
        self.title("AutoQC_ACR")
        self.geometry("+5+5")
        self.attributes("-topmost", True)
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _on_closing(self):
        if tk.messagebox.askokcancel(
            "Quit", "Are you sure you want to quit AutoQC_ACR"
        ):
            self.destroy()
            sys.exit()

    def _create_widgets(self):
        # Frames
        self.frame_global = tk.Frame(self)
        self.frame_task_checkboxes = tk.Frame(self.frame_global, bd=1, relief=tk.SOLID)

        # Labels
        self.label_title = tk.Label(
            self.frame_global,
            text="Configuration Settings",
            font=self.title_font,
            anchor="w",
        )
        self.label_in_dir = tk.Label(
            self.frame_global, text="Input directory:", font=self.text_font
        )
        self.label_out_dir = tk.Label(
            self.frame_global, text="Output directory:", font=self.text_font
        )
        self.label_select_subdirs = tk.Label(
            self.frame_global,
            text="Select subdirectories\nto process:",
            font=self.text_font,
            justify=tk.LEFT,
        )
        self.label_select_tasks = tk.Label(
            self.frame_global, text="Select tasks to run:", font=self.text_font
        )

        # Entries
        self.entry_in_dir = tk.Entry(self.frame_global, font=self.text_font)
        self.entry_out_dir = tk.Entry(self.frame_global, font=self.text_font)

        # Listboxes
        self.listbox_subdirs = tk.Listbox(
            self.frame_global,
            selectmode=tk.EXTENDED,
            font=self.text_font,
            selectforeground="black",
            selectbackground="lightgray",
            bd=1,
            relief=tk.SOLID,
        )

        # Scrollbars
        self.scrollbar_subdirs = tk.Scrollbar(
            self.frame_global,
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
                font=self.text_font,
                variable=var,
            )
            for var, task_name in zip(self.task_checkboxes_vars, AVAILABLE_TASKS)
        ]

        # Buttons
        self.button_browse_in_dir = tk.Button(
            self.frame_global,
            text="Browse",
            font=self.text_font,
            command=self._browse_in_dir,
            bd=1,
            relief=tk.SOLID,
        )
        self.button_browse_out_dir = tk.Button(
            self.frame_global,
            text="Browse",
            font=self.text_font,
            command=self._browse_out_dir,
            bd=1,
            relief=tk.SOLID,
        )
        self.button_cache_pull = tk.Button(
            self.frame_global,
            text="Pull from Cache",
            font=self.text_font,
            command=self._pull_from_cache,
            bd=1,
            relief=tk.SOLID,
        )
        self.button_run = tk.Button(
            self.frame_global,
            text="Run AutoQC_ACR",
            font=self.text_font,
            command=self.run_autoqc_acr,
            bd=1,
            relief=tk.SOLID,
        )

        # Config
        self.listbox_subdirs.config(yscrollcommand=self.scrollbar_subdirs.set)

    def _layout_widgets(self):
        self.GLOBAL_GRID_DIMS = (6, 3)
        self.ROW_PAD = (0, 15)
        self.COL_PAD = (0, 10)

        self.frame_global.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Column 0
        self.label_title.grid(
            row=0,
            column=0,
            columnspan=self.GLOBAL_GRID_DIMS[0],
            sticky="w",
            pady=self.ROW_PAD,
        )
        self.label_in_dir.grid(
            row=1, column=0, sticky="w", padx=self.COL_PAD, pady=self.ROW_PAD
        )
        self.label_out_dir.grid(
            row=2, column=0, sticky="w", padx=self.COL_PAD, pady=self.ROW_PAD
        )
        self.label_select_subdirs.grid(
            row=3, column=0, sticky="nw", padx=self.COL_PAD, pady=self.ROW_PAD
        )
        self.label_select_tasks.grid(
            row=4, column=0, sticky="nw", padx=self.COL_PAD, pady=self.ROW_PAD
        )
        self.button_run.grid(
            row=5, column=0, columnspan=self.GLOBAL_GRID_DIMS[0], sticky="ew", padx=50
        )

        # Column 1
        self.entry_in_dir.grid(
            row=1, column=1, sticky="ew", padx=self.COL_PAD, pady=self.ROW_PAD
        )
        self.entry_out_dir.grid(
            row=2, column=1, sticky="ew", padx=self.COL_PAD, pady=self.ROW_PAD
        )
        self.listbox_subdirs.grid(
            row=3, column=1, sticky="nsew", padx=self.COL_PAD, pady=self.ROW_PAD
        )
        self.frame_task_checkboxes.grid(
            row=4, column=1, sticky="nsw", padx=self.COL_PAD, pady=self.ROW_PAD
        )

        # Column 2
        self.button_browse_in_dir.grid(row=1, column=2, sticky="w", pady=self.ROW_PAD)
        self.button_browse_out_dir.grid(row=2, column=2, sticky="w", pady=self.ROW_PAD)
        self.scrollbar_subdirs.grid(row=3, column=2, sticky="nsw", pady=self.ROW_PAD)

        # Other layout
        for task_checkbox in self.task_checkboxes:
            task_checkbox.pack(anchor="w", padx=self.COL_PAD)

    def _configure_grid(self):
        self.frame_global.columnconfigure(0, weight=0, minsize=0)
        self.frame_global.columnconfigure(1, weight=1, minsize=400)
        self.frame_global.columnconfigure(2, weight=0, minsize=0)

        self.frame_global.rowconfigure(0, weight=0, minsize=0)
        self.frame_global.rowconfigure(1, weight=0, minsize=0)
        self.frame_global.rowconfigure(2, weight=0, minsize=50)
        self.frame_global.rowconfigure(3, weight=1, minsize=0)

    def _populate_entry_widget(self, entry_widget: tk.Entry, text: str):
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, text)
        entry_widget.icursor(tk.END)
        entry_widget.xview_moveto(1)

    def _populate_listbox_with_subdirs(self, listbox: tk.Listbox, in_dir: Path):
        listbox.delete(0, tk.END)
        for subdir in in_dir.iterdir():
            if subdir.is_dir():
                listbox.insert(tk.END, subdir.name)

    def _browse_in_dir(self):
        in_dir = Path(filedialog.askdirectory(title="Select Input Directory"))
        if in_dir:
            self._populate_entry_widget(self.entry_in_dir, str(in_dir.resolve()))
            self._populate_entry_widget(
                self.entry_out_dir,
                str((in_dir.parent / "AutoQC_ACR_Output").resolve()),
            )

            self._sort_loose_dicoms(in_dir)
            self._populate_listbox_with_subdirs(
                self.listbox_subdirs,
                in_dir,
            )

    def _browse_out_dir(self):
        out_dir = Path(filedialog.askdirectory(title="Select Output Directory"))
        if out_dir:
            self._populate_entry_widget(self.entry_out_dir, str(out_dir.resolve()))

    def _sort_loose_dicoms(self, in_dir: Path):
        files = list(in_dir.glob("*"))
        dcms = []
        ModalListProcessor(
            master=self,
            task_name="Please wait, checking for loose DICOMs!",
            objs_to_process=files,
            processing_func=lambda x: dcms.append(x) if quick_check_dicom(x) else None,
        )

        if dcms:

            def sort_single_dcm(dcm: Path):
                target_folder = in_dir / pydicom.dcmread(dcm).SeriesDescription
                target_folder.mkdir(exist_ok=True)
                dcm.rename(target_folder / dcm.name)

            ModalListProcessor(
                master=self,
                task_name="Please wait, sorting loose DICOMs!",
                objs_to_process=dcms,
                processing_func=sort_single_dcm,
            )

    def _pull_from_cache(self):
        pass

    def run_autoqc_acr(self):
        # Check for valid in_dir selection
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

        config = {
            "in_dir": str(in_dir.resolve()),
            "out_dir": str(out_dir.resolve()),
            "in_subdirs": [str(x.resolve()) for x in in_subdirs],
            "out_subdirs": [str(x.resolve()) for x in out_subdirs],
            "tasks_to_run": tasks_to_run,
        }

        task_runner = TaskRunner(
            self, tasks_to_run, in_dir, out_dir, in_subdirs, out_subdirs
        )
        task_runner.run()
