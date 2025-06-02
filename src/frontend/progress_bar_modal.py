import tkinter as tk
from tkinter import ttk


class ProgressBarModal(tk.Toplevel):
    """Modal window to display progress of a task whilst grabbing focus."""

    def __init__(
        self,
        master: tk.Tk,
        task_name: str,
    ):
        """Initialises ProgressBarModal.

        Args:
            master (tk.Tk): Root window of the application
            task_name (str): Task name to display in the modal
        """
        super().__init__(master)
        self.title("Progress: 0% Complete")
        self.task_name = task_name

        self.progress_var = tk.DoubleVar()
        self.progress_var.set(0.0)

        self._create_widgets()
        self._layout_widgets()

        self.transient(master)
        self.grab_set()

        self.after(5, self._centre_on_master, master)

    def _create_widgets(self):
        """Creates widgets for modal window."""
        self.label = tk.Label(self, text=self.task_name)
        self.progress_bar = ttk.Progressbar(
            self, variable=self.progress_var, maximum=100
        )

    def _layout_widgets(self):
        """Lays out widgets in modal window."""
        self.label.pack(pady=10, padx=25)
        self.progress_bar.pack(fill=tk.X, padx=20, pady=10)

    def _centre_on_master(self, master):
        """Positions the modal window in the centre of the master window."""
        x_to_set = (
            master.winfo_x() + master.winfo_width() // 2 - self.winfo_width() // 2
        )
        y_to_set = (
            master.winfo_y() + master.winfo_height() // 2 - self.winfo_height() // 2
        )
        self.geometry(f"+{x_to_set}+{y_to_set}")

    def add_progress(self, value: float):
        """Adds progress to the progress bar widget.

        Args:
            value (float): Value to add to current progress bar state.
                Should be between 0 and 100.
        """
        current_progress = self.progress_var.get()
        new_progress = current_progress + value

        # Ensure progress does not exceed the maximum
        if new_progress > 100:
            new_progress = 100

        self.progress_var.set(new_progress)
        self.title(f"Progress: {int(new_progress)}% Complete")
        self.update_idletasks()
