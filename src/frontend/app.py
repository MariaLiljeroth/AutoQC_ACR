"""
app.py

This script defines the App class.
App inherits from tk.Tk and acts as the toplevel root window for the page-based GUI application.
App can swap what its displaying by switching the currently displayed "page" (tk.Frame subclass).
Page switching and other triggers are initiatied through the global multiprocessing queue.

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
    GUI works through switching "pages" (subclasses of tk.Frame)
    """

    # Padding around the edge of displayed pages.
    PAD_PAGE = 20

    # Maps the names of pages to their associated page subclasses.
    PAGE_MAP = dict(zip(["CONFIG", "TASKRUNNER"], [PageConfig, PageTaskRunner]))

    def __init__(self):
        """Initialises the App class, displays configuration page
        and initialises periodic queue checking for queue triggers.

        Instance attributes:
            pages (dict): Dictionary of page instances. Page names are mapped directly to existing instances of associated page classes.
            current_page (tk.Frame): Currently displayed page.
        """

        # Set properites of self (tk.Tk)
        super().__init__()
        self.geometry(f"+5+5")
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.title("AutoQC_ACR")

        # initialise class to store general app state.
        self.app_state = AppState()

        # Initialise dictionary of page name -> instance mappings.
        self.pages = {}

        # To begin with, show configuration page.
        self.show_page(PageConfig)

        # Initialise queue checking.
        self._check_queue()

    def show_page(self, page_class: tk.Frame):
        """Takes a page class and checks whether an instance
        already exists in self.pages. If it exists, the frontend displays this page.
        Otherwise, an instance of the page class is created and displayed.

        Args:
            page_class (tk.Frame): Page to display.
        """

        # Set self to be resizible so can adjust to fit dimensions of newly swapped in page.
        self.resizable(True, True)

        # Get class name of page that is trying to be swapped in
        page_name = page_class.__name__

        # If instance of that particular page class does not exist in self.pages,
        # instance is created, stored in self.pages and packed.
        if page_name not in self.pages:

            # Create instance of page class that is trying to be swapped in
            page = page_class(self, self.app_state)

            # Track newly created instance in self.pages.
            self.pages[page_name] = page

            # Pack instance into self with padding.
            page.pack(fill=tk.BOTH, expand=True, padx=self.PAD_PAGE, pady=self.PAD_PAGE)
            if hasattr(self, "current_page"):
                self.current_page.pack_forget()

        # else matching page is selected from self.pages.
        else:
            page = self.pages[page_name]

        # Set self.current_page to newly swapped in page and raise page to be on top of others
        self.current_page = page
        self.after(50, page.tkraise)

        # Lock dimensions of self now that self will have adjusted to fit new page.
        self.after(100, lambda: self.resizable(False, False))

    def _check_queue(self):
        """Initialises queue checking for triggers. App-level triggers are immediately handled.
        Otherwise, trigger is passed to self.current_page for local processing within that object.
        """

        # Try and get trigger from queue
        try:
            while True:

                # Get trigger
                trigger = get_queue().get_nowait()

                # Handle request to switch pages
                if trigger.ID == "SWITCH_PAGE":

                    # Get name of page to swap in.
                    page_name = trigger.data

                    # Get class associated with new page's name
                    page_class = self.PAGE_MAP[page_name]

                    # Swap in new page.
                    self.show_page(page_class)

                # Handles application quitting trigger.
                elif trigger.ID == "QUIT_APPLICATION":

                    # Show messagebox that the application has finished running and quit GUI and sys.
                    messagebox.showinfo(
                        "Quit", "AutoQC_ACR has finished running. Check the results!"
                    )
                    self.destroy()
                    sys.exit()

                # Other triggers passed down to current visible page, to be handled locally
                # (trigger not relevant for here).
                else:
                    self.current_page.handle_task_request(trigger)

        # Pass if queue has no trigger.
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
