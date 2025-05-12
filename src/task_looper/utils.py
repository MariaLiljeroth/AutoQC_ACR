from collections import defaultdict
from thefuzz import fuzz
from pathlib import Path
import pydicom

import tkinter as tk
from tkinter import ttk
from typing import Callable
import threading


def nested_dict():
    return defaultdict(nested_dict)


def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        return {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d


def substring_matcher(string: str, strings_to_search: list[str]) -> str:
    best_match = sorted(
        strings_to_search,
        key=lambda x: fuzz.ratio(x.lower(), string.lower()),
        reverse=True,
    )[0]
    return best_match


def quick_check_dicom(file: Path) -> bool:
    try:
        with file.open("rb") as f:
            f.seek(128)
            if f.read(4) == b"DICM":
                return True
    except:
        pass

    try:
        pydicom.dcmread(file, stop_before_pixels=True)
        return True
    except:
        return False


class ModalListProcessor(tk.Toplevel):

    def __init__(
        self,
        master: tk.Tk,
        task_name: str,
        objs_to_process: list,
        processing_func: Callable,
    ):
        super().__init__(master)
        self.title("Progress: 0% Complete")
        self.task_name = task_name
        self.objs_to_process = objs_to_process
        self.processing_func = processing_func

        self.progress_var = tk.DoubleVar()
        self.progress_var.set(0.0)

        self._create_widgets()
        self._layout_widgets()

        self.transient(master)
        self.grab_set()

        self.after(5, self._centre_on_master, master)

        threading.Thread(target=self._run_task).start()
        master.wait_window(self)

    def _create_widgets(self):
        self.label = tk.Label(self, text=self.task_name)
        self.progress_bar = ttk.Progressbar(
            self, variable=self.progress_var, maximum=100
        )

    def _layout_widgets(self):
        self.label.pack(pady=10, padx=25)
        self.progress_bar.pack(fill=tk.X, padx=20, pady=10)

    def _centre_on_master(self, master):
        x_to_set = (
            master.winfo_x() + master.winfo_width() // 2 - self.winfo_width() // 2
        )
        y_to_set = (
            master.winfo_y() + master.winfo_height() // 2 - self.winfo_height() // 2
        )
        self.geometry(f"+{x_to_set}+{y_to_set}")

    def _run_task(self):
        total = len(self.objs_to_process)
        for i, obj in enumerate(self.objs_to_process):
            self.processing_func(obj)
            self.after(0, self._update_progress, (i + 1) / total * 100)
        self.after(0, self.destroy)

    def _update_progress(self, value: float):
        self.progress_var.set(value)
        self.title(f"Progress: {int(value)}% Complete")
        self.update_idletasks()
