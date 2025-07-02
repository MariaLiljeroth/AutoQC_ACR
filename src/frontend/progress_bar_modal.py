"""
progress_bar_modal.py

This script defines the ProgressBarModal class, which can be initialised to create a modal progress bar.
This progress bar is needed for tasks that cannot run in the background, i.e. those that must be completed
before the user continues within the GUI (e.g. DICOM sorting).

"""

import tkinter as tk
from tkinter import ttk


class ProgressBarModal(tk.Toplevel):
    """Modal window to display progress of a compulsory task whilst grabbing focus
    to prevent user interacting further with the GUI. Inherits from tk.Toplevel.
    """

    def __init__(
        self,
        master: tk.Tk,
        task_name: str,
    ):
        """Initialises ProgressBarModal by configuring window settings, creating
        and laying out widgets and grabbing the GUI focus.

        Args:
            master (tk.Tk): Root window of the application to create the modal window on top of.
            task_name (str): Task name to display in the modal window for the user's information.
        """
        super().__init__(master)

        # Set window title to reflect percentage completion - 0%.
        self.title("Progress: 0% Complete")

        # Name of task to complete saved as instance attribute for convenience.
        self.task_name = task_name

        # Create attribute to store progress bar progress as a double.
        self.progress_var = tk.DoubleVar()
        self.progress_var.set(0.0)

        # Create widgets for GUI.
        self._create_widgets()

        # Layout widgets for GUI.
        self._layout_widgets()

        # Link modal window to master, the root window.
        self.transient(master)

        # Grab GUI focus to prevent user from interacting with GUI whilst task is running.
        self.grab_set()

        # Centre modal window on master, for visuals.
        self.after(5, self._centre_on_master, master)

    def _create_widgets(self):
        """Creates all widgets to be contained within self."""
        # Label showing the user what task is running.
        self.label = tk.Label(self, text=self.task_name)

        # Progress bar to show the user how far along the task has progressed.
        self.progress_bar = ttk.Progressbar(
            self, variable=self.progress_var, maximum=100
        )

    def _layout_widgets(self):
        """Lays out created widgets within self."""
        # Use simple packing to lay out, with padding.
        self.label.pack(pady=10, padx=25)
        self.progress_bar.pack(fill=tk.X, padx=20, pady=10)

    def _centre_on_master(self, master):
        """Positions the modal window in the centre of the master window, for visual appeal."""
        # Calculate x and y values of self that are required for centering on master window
        x_to_set = (
            master.winfo_x() + master.winfo_width() // 2 - self.winfo_width() // 2
        )
        y_to_set = (
            master.winfo_y() + master.winfo_height() // 2 - self.winfo_height() // 2
        )

        # Set position of window to calculated (x, y)
        self.geometry(f"+{x_to_set}+{y_to_set}")

    def add_progress(self, value: float):
        """Adds a set amount of progress to the progress bar widget.

        Args:
            value (float): Value to add to current progress bar state.
                Should be between 0 and 100.
        """

        # Get current state of progress bar.
        current_progress = self.progress_var.get()

        # Variable to define new progress after modification.
        new_progress = current_progress + value

        # Ensure progress bar does not exceed 100% for realism.
        if new_progress > 100:
            new_progress = 100

        # Update progress bar variable with new value.
        self.progress_var.set(new_progress)

        # Set title to new progress value and update GUI.
        self.title(f"Progress: {int(new_progress)}% Complete")
        self.update_idletasks()
