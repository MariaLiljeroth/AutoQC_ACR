import sys
import traceback

import tkinter as tk
from tkinter import messagebox

from shared.queueing import get_queue
from queue import Empty

from frontend.frames.frame_config import FrameConfig
from frontend.frames.frame_task_runner import FrameTaskRunner


class App(tk.Tk):
    """Centralised frontend app class for AutoQC_ACR.
    GUI works through switching subclasses of tk.Frame
    """

    PAD_FRAME = 20
    FRAME_MAP = dict(zip(["CONFIG", "TASKRUNNER"], [FrameConfig, FrameTaskRunner]))

    def __init__(self):
        """Initialises the App class, displays configuration frame
        and initialises queue checking.
        """
        super().__init__()
        self.geometry(f"+5+5")
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.title("AutoQC_ACR")

        self.frames = {}
        self.show_frame(FrameConfig)

        self._check_queue()

    def show_frame(self, frame_class: tk.Frame, *args_to_pass):
        """Takes a tk.Frame subclass and checks whether an instance
        already exists in self.frames. If it exists, GUI displays this frame.
        Otherwise, an instance of the frame is created and displayed.

        Args:
            frame_class (tk.Frame): Frame to display.
        """
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
        """Initialises queue checking for events. App-level events are immediately handled.
        Otherwise, event is passed to self.current_frame for local processing.
        """
        try:
            while True:
                event = get_queue().get_nowait()
                if event[0] == "SWITCH_FRAME":
                    # Frame switching event.
                    # Possible to pass args from caller frame to new frame through event[2]
                    frame_name = event[1]
                    if len(event) >= 3:
                        args_to_pass = event[2]
                    else:
                        args_to_pass = []
                    frame_class = self.FRAME_MAP[frame_name]
                    self.show_frame(frame_class, *args_to_pass)
                elif event[0] == "QUIT_APPLICATION":
                    self.destroy()
                    sys.exit()
                else:
                    # Pass event to current visible frame
                    self.current_frame.handle_event(event)
        except Empty:
            pass
        except Exception as e:
            traceback.print_exc()
            print("Error during queue check:", e)
        finally:
            self.after(100, self._check_queue)

    def _on_closing(self):
        """Handles manual quitting of application."""
        if messagebox.askokcancel("Quit", "Are you sure you want to quit AutoQC_ACR?"):
            self.destroy()
            sys.exit()
