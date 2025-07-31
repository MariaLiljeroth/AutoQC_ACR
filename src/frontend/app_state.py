"""
app_state.py

Script defines AppState class. This class is instantiated once in frontend/app.py and
passed to all subsequent page objects. Instance attributes represent useful information
about the state of the app. These attributes can be accessed by all pages.

Written by Nathan Crossley, 2025.
"""


class AppState:
    """AppState class.
    This class is instantiated once in frontend/app.py
    and passed to all subsequent page objects. Instance
    attributes represent useful informationabout the state
    of the app. These attributes can be accessed by all pages.
    """

    def __init__(self):
        """Initialises AppState, setting all attributes to None."""

        self.in_dir = None
        self.out_dir = None
        self.in_subdirs = None
        self.out_subdirs = None
        self.tasks_to_run = None
        self.baselines = None
        self.log_path = None
        self.results = None
