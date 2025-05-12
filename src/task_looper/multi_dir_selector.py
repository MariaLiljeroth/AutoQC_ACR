import sys
import tkinter as tk
from tkinter import messagebox
from pathlib import Path


class MultiDirSelector(tk.Tk):
    width = 500
    height = 250
    pad_large = 20
    pad_small = 10
    bg_color = "#ADD8E6"

    def __init__(self, dirs: list[Path]) -> list[Path]:
        super().__init__()
        self.title("Please select all subdirectories to process")
        self.geometry(f"{self.width}x{self.height}")
        self.attributes("-topmost", True)
        self.dirs = dirs
        self.configure(bg=self.bg_color)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.main_frame = tk.Frame(self, bg=self.bg_color)
        self.main_frame.pack(
            fill=tk.BOTH, expand=True, padx=self.pad_large, pady=self.pad_large
        )

        self.listbox = tk.Listbox(
            self.main_frame, selectmode=tk.MULTIPLE, bd=1, relief=tk.SOLID
        )
        self.listbox.grid(row=0, column=0, stick="nsew")

        scrollbar = tk.Scrollbar(
            self.main_frame,
            orient=tk.VERTICAL,
            bd=1,
            relief=tk.SOLID,
            command=self.listbox.yview,
        )
        scrollbar.grid(row=0, column=1, sticky="nsw")

        self.listbox.config(yscrollcommand=scrollbar.set)

        for dir in self.dirs:
            self.listbox.insert(tk.END, dir.name)

        self.done_button = tk.Button(
            self.main_frame,
            text="Select Directories",
            width=20,
            height=2,
            bd=1,
            relief=tk.SOLID,
            command=self.get_selection,
        )
        self.done_button.grid(row=1, column=0, pady=(self.pad_small, 0))

        self.init_grid()

    def init_grid(self):
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=0)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=0)

    def get_selection(self):
        self.selected_dirs = [self.dirs[i] for i in self.listbox.curselection()]
        self.unselected_dirs = [
            dir for dir in self.dirs if dir not in self.selected_dirs
        ]
        if not self.selected_dirs:
            messagebox.showwarning("No Selection", "No subdirectories selected.")
            return
        self.quit()

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Are you sure you want to quit MR analysis?"):
            self.destroy()
            sys.exit()

    def run(self):
        self.mainloop()
        self.destroy()
        return self.selected_dirs, self.unselected_dirs
