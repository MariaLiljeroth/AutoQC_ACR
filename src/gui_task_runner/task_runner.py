from pathlib import Path
import tkinter as tk
from tkinter import ttk

import multiprocessing as mp
import threading
from queue import Empty

from utils import nested_dict, defaultdict_to_dict, substring_matcher
from context import AVAILABLE_TASKS, EXPECTED_ORIENTATIONS, EXPECTED_COILS

from hazenlib.tasks.acr_slice_thickness import ACRSliceThickness
from hazenlib.tasks.acr_snr import ACRSNR
from hazenlib.tasks.acr_geometric_accuracy import ACRGeometricAccuracy
from hazenlib.tasks.acr_uniformity import ACRUniformity
from hazenlib.tasks.acr_spatial_resolution import ACRSpatialResolution
from hazenlib.utils import get_dicom_files

TASK_STR_TO_CLASS = dict(
    zip(
        AVAILABLE_TASKS,
        [
            ACRSliceThickness,
            ACRSNR,
            ACRGeometricAccuracy,
            ACRUniformity,
            ACRSpatialResolution,
        ],
    )
)


class TaskRunner(tk.Toplevel):

    text_font = ("Segoe UI", 11)
    title_font = ("Segoe UI", 14, "bold", "underline")

    def __init__(
        self,
        master: tk.Tk,
        tasks_to_run: list[str],
        in_dir: Path,
        out_dir: Path,
        in_subdirs: list[Path],
        out_subdirs: list[Path],
    ):
        super().__init__(master)
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.master = master
        self.tasks_to_run = tasks_to_run
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.in_subdirs = in_subdirs
        self.out_subdirs = out_subdirs

        self._create_widgets()
        self._layout_widgets()
        self._configure_grid()

        self.transient(master)
        self.grab_set()

        self.after(5, self._centre_on_master, master)
        # self.after(20, master.withdraw)

    def _create_widgets(self):
        # Frames
        self.frame_global = tk.Frame(self)

        # Labels
        self.label_title = tk.Label(
            self.frame_global, text="Task Progress", font=self.title_font
        )
        self.label_prep_progress = tk.Label(
            self.frame_global,
            text="Preparing to run:",
            font=self.text_font,
            anchor="center",
        )
        self.label_tasks_progress = tk.Label(
            self.frame_global,
            text="Task progress:",
            font=self.text_font,
            anchor="center",
        )

        # Progress bars
        self.progress_bar_prep = ttk.Progressbar(
            self.frame_global, mode="indeterminate"
        )
        self.progress_bar_tasks = ttk.Progressbar(self.frame_global, maximum=100)

    def _layout_widgets(self):
        self.GLOBAL_GRID_DIMS = (4, 2)
        self.ROW_PAD = (0, 15)
        self.COL_PAD = (0, 10)

        self.frame_global.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Column 0
        self.label_title.grid(
            row=0, column=0, columnspan=2, sticky="w", pady=self.ROW_PAD
        )
        self.label_prep_progress.grid(
            row=1,
            column=0,
            sticky="w",
            padx=self.COL_PAD,
            pady=self.ROW_PAD,
        )
        self.label_tasks_progress.grid(
            row=2,
            column=0,
            sticky="w",
            padx=self.COL_PAD,
            pady=self.ROW_PAD,
        )

        # Column 1
        self.progress_bar_prep.grid(row=1, column=1, sticky="nsew", pady=self.ROW_PAD)
        self.progress_bar_tasks.grid(row=2, column=1, sticky="nsew", pady=self.ROW_PAD)

    def _configure_grid(self):
        self.frame_global.grid_rowconfigure(0, weight=0)
        self.frame_global.grid_rowconfigure(1, weight=0)
        self.frame_global.grid_rowconfigure(2, weight=0)

        self.frame_global.grid_columnconfigure(0, weight=0)
        self.frame_global.grid_columnconfigure(1, weight=1, minsize=300)

    def _centre_on_master(self, master):
        x_to_set = (
            master.winfo_x() + master.winfo_width() // 2 - self.winfo_width() // 2
        )
        y_to_set = (
            master.winfo_y() + master.winfo_height() // 2 - self.winfo_height() // 2
        )
        self.geometry(f"+{x_to_set}+{y_to_set}")

    def _on_closing(self):
        if tk.messagebox.askokcancel(
            "Quit", "Are you sure you want to quit AutoQC_ACR?"
        ):
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()
            if hasattr(self, "pool"):
                self.pool.terminate()
                self.pool.join()

            self.destroy()

    def run(self):
        self.progress_bar_prep.start(25)
        threading.Thread(target=self._setup_multiprocessing).start()

    def _setup_multiprocessing(self):
        self.queue = mp.Manager().Queue()

        self.process = mp.Process(
            target=self._run_tasks,
            args=(self.queue, self.in_subdirs, self.out_subdirs, self.tasks_to_run),
        )
        self.process.start()
        self._check_queue()

    @classmethod
    def _run_tasks(cls, queue, in_subdirs, out_subdirs, tasks_to_run):
        cpu_cores = mp.cpu_count()
        args = [
            (in_subdir, out_subdir, task, queue)
            for in_subdir, out_subdir in zip(in_subdirs, out_subdirs)
            for task in tasks_to_run
        ]

        cls.pool = mp.Pool(processes=cpu_cores // 2)
        try:
            queue.put("STOP_PREP_BAR")
            results = cls.pool.starmap(cls._run_solo_task_on_folder, args)

        finally:
            cls.pool.close()
            cls.pool.join()
            queue.put(("RESULTS"), results)
            queue.put("ALL_TASKS_COMPLETE")

    #     # for in_subdir, out_subdir, task in args:
    #     #     result = self._run_solo_task_on_folder(in_subdir, out_subdir, task)
    #     #     coil = substring_matcher(in_subdir.name, EXPECTED_COILS)
    #     #     orientation = substring_matcher(in_subdir.name, EXPECTED_ORIENTATIONS)
    #     #     self.results[task][coil][orientation] = result

    @staticmethod
    def _run_solo_task_on_folder(in_subdir, out_subdir, task, queue):
        task_obj = TASK_STR_TO_CLASS[task](
            input_data=get_dicom_files(in_subdir.resolve()),
            report_dir=out_subdir.resolve(),
            report=True,
            MediumACRPhantom=True,
        )
        result = task_obj.run()
        queue.put("UPDATE_TASK_BAR")

        return result

    def _check_queue(self):
        """Check the queue for updates and handle them."""
        try:
            while True:
                message = self.queue.get_nowait()  # Non-blocking get
                if message == "STOP_PREP_BAR":
                    self.progress_bar_prep.stop()
                    self.progress_bar_prep.config(mode="determinate", maximum=100)
                    self.progress_bar_prep["value"] = 100

                elif message == "UPDATE_TASK_BAR":
                    total_tasks = len(self.tasks_to_run) * len(self.in_subdirs)
                    self.progress_bar_tasks["value"] += 1 / total_tasks * 100

                elif message == "ALL_TASKS_COMPLETE":
                    self.process.join()
                    return

                elif isinstance(message, tuple) and message[0] == "RESULTS":
                    self.results = message[1]

        except Empty:
            # No messages in the queue, check again later
            pass
        finally:
            # Schedule the next queue check
            self.after(100, self._check_queue)
