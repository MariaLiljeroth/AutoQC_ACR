import sys

import tkinter as tk
from tkinter import messagebox

from shared.global_queue import get_queue
from queue import Empty

from frontend.frames.frame_config import FrameConfig
from frontend.frames.frame_task_runner import FrameTaskRunner


class App(tk.Tk):

    PAD_FRAME = 20
    FRAME_MAP = {"FRAME_CONFIG": FrameConfig, "FRAME_TASKRUNNER": FrameTaskRunner}

    def __init__(self):
        super().__init__()
        self.geometry(f"+5+5")
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.title("AutoQC_ACR")

        self.frames = {}
        self.show_frame(FrameConfig)

        self._check_queue()

    def show_frame(self, frame_class: tk.Frame, *args_to_pass):
        frame_name = frame_class.__name__
        if frame_name not in self.frames:
            frame = frame_class(self, *args_to_pass)
            self.frames[frame_name] = frame
            frame.pack(
                fill=tk.BOTH, expand=True, padx=self.PAD_FRAME, pady=self.PAD_FRAME
            )
            if hasattr(self, "current_frame"):
                self.current_frame.pack_forget()

        else:
            frame = self.frames[frame_name]
        self.current_frame = frame
        frame.tkraise()

    def _check_queue(self):
        try:
            while True:
                event = get_queue().get_nowait()
                print(event)
                if isinstance(event, tuple) and event[0] == "SWITCH_FRAME":
                    frame_name = event[1]
                    args_to_pass = event[2]
                    frame_class = self.FRAME_MAP[frame_name]
                    self.show_frame(frame_class, *args_to_pass)
                else:
                    self.current_frame.handle_event(event)
        except Empty:
            pass
        except Exception as e:
            print("Error during queue check:", e)
        finally:
            self.after(100, self._check_queue)

    def _on_closing(self):
        if messagebox.askokcancel("Quit", "Are you sure you want to quit AutoQC_ACR?"):
            self.destroy()
            sys.exit()
