"""
app.py

This script defines the App class.
App inherits from tk.Tk and acts as the toplevel root window for the page-based GUI application.
App can swap what its displaying by switching the currently displayed tk.Frame subclass (page).
Page swapping and other triggers are initiatied through the global multiprocessing queue.

Written by Nathan Crossley 2025

"""

import sys
import traceback

import tkinter as tk
from tkinter import messagebox

from queue import Empty
from src.shared.queueing import get_queue

from src.frontend.app_state import AppState
from src.frontend.pages.page_config import PageConfig
from src.frontend.pages.page_task_runner import PageTaskRunner


class App(tk.Tk):
    """Centralised frontend app class for AutoQC_ACR.
    GUI works through switching subclasses of tk.Frame (pages)
    """

    # Padding around the edge of displayed pages.
    PAD_PAGE = 20

    # Maps the names of pages to their associated page subclasses.
    PAGE_MAP = dict(zip(["CONFIG", "TASKRUNNER"], [PageConfig, PageTaskRunner]))

    def __init__(self):
        """Initialises the App class, displays configuration page
        and initialises queue checking.

        Instance attributes:
            pages (dict): Dictionary of page instances. Page names are mapped directly to existing instances of associated page classes.
            current_page (tk.Frame): Currently displayed page.
        """

        # Set properites of self (tk.Tk)
        super().__init__()
        self.geometry(f"+5+5")
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.title("AutoQC_ACR")
        self.app_state = AppState()

        # Initialise dictionary of page name -> instance mappings.
        self.pages = {}

        # To begin with, show configuration page.
        self.show_page(PageConfig)

        # Initialise queue checking.
        self._check_queue()

    def show_page(self, page_class: tk.Frame, *args_to_pass: any):
        """Takes a tk.Frame subclass (a page) and checks whether an instance
        already exists in self.pages. If it exists, GUI displays this page.
        Otherwise, an instance of the page is created and displayed.

        Args:
            page_class (tk.Frame): Page to display.
            args_to_pass(any): Additional arguments to pass to the page when switching.
        """

        # Set self to be resizible so can adjust to fit dimensions of newly swapped in page.
        self.resizable(True, True)

        # Get class name of page that is trying to be swapped in
        page_name = page_class.__name__

        # If instance of that particular page class does not exist in self.pages, instance is created and displayed.
        if page_name not in self.pages:

            # Create instance of page class that is trying to be swapped in
            page = page_class(self, *args_to_pass)

            # Track newly created instance in self.pages.
            self.pages[page_name] = page

            # Pack instance into self with padding.
            page.pack(fill=tk.BOTH, expand=True, padx=self.PAD_PAGE, pady=self.PAD_PAGE)
            if hasattr(self, "current_page"):
                self.current_page.pack_forget()

        else:
            page = self.pages[page_name]

        # Set current_page attribute to newly swapped in page.
        self.current_page = page

        # Raise newly swapped in page to be on top of others.
        self.after(50, page.tkraise)

        # Lock dimensions of self now that newly swapped in page has expanded.
        self.after(100, lambda: self.resizable(False, False))

    def _check_queue(self):
        """Initialises queue checking for events. App-level events are immediately handled.
        Otherwise, event is passed to self.current_page for local processing within that object.
        """

        # Try and get event message in queue
        try:
            while True:

                # Get event message
                event = get_queue().get_nowait()

                # Handles page switching event. Additional args can be pased from caller page to new page through event[2].
                if event[0] == "SWITCH_PAGE":

                    # Get name of page to swap in.
                    page_name = event[1]

                    # Get args to pass to new page.
                    if len(event) >= 3:
                        args_to_pass = event[2]
                    else:
                        args_to_pass = []

                    # Get class associated with new page's name
                    page_class = self.PAGE_MAP[page_name]

                    # Swap in new page.
                    self.show_page(page_class, *args_to_pass)

                # Handles application quitting event and end of AutoQC_ACR runtime.
                elif event[0] == "QUIT_APPLICATION":

                    # Show messagebox that the application has finished running and quit GUI and sys.
                    messagebox.showinfo(
                        "Quit", "AutoQC_ACR has finished running. Check the results!"
                    )
                    self.destroy()
                    sys.exit()

                # Other events passed down to current visible page, to be handled locally (event not relevant for here) due to separation of concerns.
                else:
                    self.current_page.handle_event(event)

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
