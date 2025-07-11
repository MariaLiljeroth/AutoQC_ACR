"""
app.py

This script defines the App class.
App inherits from tk.Tk and acts as the toplevel root window for the frame-based GUI application.
App can swap what its displaying by switching the currently displayed tk.Frame subclass.
Frame swapping and other triggers are initiatied through the global multiprocessing queue.

Written by Nathan Crossley 2025

"""

import sys
import traceback

import tkinter as tk
from tkinter import messagebox

from queue import Empty
from src.shared.queueing import get_queue

from src.frontend.frames.frame_config import FrameConfig
from src.frontend.frames.frame_task_runner import FrameTaskRunner


class App(tk.Tk):
    """Centralised frontend app class for AutoQC_ACR.
    GUI works through switching subclasses of tk.Frame
    """

    # Padding around the edge of displayed frames.
    PAD_FRAME = 20

    # Maps the names of frames to their associated tk.Frame subclasses.
    FRAME_MAP = dict(zip(["CONFIG", "TASKRUNNER"], [FrameConfig, FrameTaskRunner]))

    def __init__(self):
        """Initialises the App class, displays configuration frame
        and initialises queue checking.

        Instance attributes:
            frames (dict): Dictionary of frame instances. Frame names are mapped directly to existing instances of associated tk.Frame subclasses.
            current_frame (tk.Frame): Currently displayed frame.
        """

        # Set properites of self (tk.Tk)
        super().__init__()
        self.geometry(f"+5+5")
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.title("AutoQC_ACR")

        # Initialise dictionary of frame name -> instance mappings.
        self.frames = {}

        # To begin with, show configuration frame.
        self.show_frame(FrameConfig)

        # Initialise queue checking.
        self._check_queue()

    def show_frame(self, frame_class: tk.Frame, *args_to_pass: any):
        """Takes a tk.Frame subclass and checks whether an instance
        already exists in self.frames. If it exists, GUI displays this frame.
        Otherwise, an instance of the frame is created and displayed.

        Args:
            frame_class (tk.Frame): Frame to display.
            args_to_pass(any): Additional arguments to pass to the frame when switching.
        """

        # Set self to be resizible so can adjust to fit dimensions of newly swapped in frame.
        self.resizable(True, True)

        # Get class name of frame that is trying to be swapped in
        frame_name = frame_class.__name__

        # If instance of that particular frame class does not exist in self.frames, instance is created and displayed.
        if frame_name not in self.frames:

            # Create instance of frame class that is trying to be swapped in
            frame = frame_class(self, *args_to_pass)

            # Track newly created instance in self.frames.
            self.frames[frame_name] = frame

            # Pack instance into self with padding.
            frame.pack(
                fill=tk.BOTH, expand=True, padx=self.PAD_FRAME, pady=self.PAD_FRAME
            )
            if hasattr(self, "current_frame"):
                self.current_frame.pack_forget()

        else:
            frame = self.frames[frame_name]

        # Set current_frame attribute to newly swapped in frame.
        self.current_frame = frame

        # Raise newly swapped in frame to be on top of others.
        self.after(50, frame.tkraise)

        # Lock dimensions of self now that newly swapped in frame has expanded.
        self.after(100, lambda: self.resizable(False, False))

    def _check_queue(self):
        """Initialises queue checking for events. App-level events are immediately handled.
        Otherwise, event is passed to self.current_frame for local processing within that object.
        """

        # Try and get event message in queue
        try:
            while True:

                # Get event message
                event = get_queue().get_nowait()

                # Handles frame switching event. Additional args can be pased from caller frame to new frame through event[2].
                if event[0] == "SWITCH_FRAME":

                    # Get name of frame to swap in.
                    frame_name = event[1]

                    # Get args to pass to new frame.
                    if len(event) >= 3:
                        args_to_pass = event[2]
                    else:
                        args_to_pass = []

                    # Get class associated with new frame's name
                    frame_class = self.FRAME_MAP[frame_name]

                    # Swap in new frame.
                    self.show_frame(frame_class, *args_to_pass)

                # Handles application quitting event and end of AutoQC_ACR runtime.
                elif event[0] == "QUIT_APPLICATION":

                    # Show messagebox that the application has finished running and quit GUI and sys.
                    messagebox.showinfo(
                        "Quit", "AutoQC_ACR has finished running. Check the results!"
                    )
                    self.destroy()
                    sys.exit()

                # Other events passed down to current visible frame, to be handled locally (event not relevant for here) due to separation of concerns.
                else:
                    self.current_frame.handle_event(event)

        # Pass if queue has no message
        except Empty:
            pass

        # Handle any queue errors by printing traceback for easy debugging.
        except Exception as e:
            traceback.print_exc()
            print("Error during queue check:", e)

        # Recursively check the queue whilst the app mainloop is running.
        finally:
            self.after(100, self._check_queue)

    def _on_closing(self):
        """Handles manual quitting of application."""

        # Messagebox asks if user wants to quit and appropriate action taken.
        if messagebox.askokcancel("Quit", "Are you sure you want to quit AutoQC_ACR?"):
            self.destroy()
            sys.exit()
