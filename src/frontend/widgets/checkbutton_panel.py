"""
checkbutton_panel.py

Script defines subclass of tk.Frame to represent an array of checkbuttons
arranged vertically in one panel.

Written by Nathan Crossley, 2025.
"""

import tkinter as tk

from src.frontend.settings import FONT_TEXT


class CheckbuttonPanel(tk.Frame):
    """Subclass of tk.Frame to represent an array of checkbuttons
    arranged vertically in one "panel".
    """

    # determine default bool state for checkbutton.
    DEFAULT_CHECKBUTTON_STATE = True

    # define default kwargs for all tk widgets. User can override.
    DEFAULT_KWARGS = {
        "checkbutton": {"font": FONT_TEXT, "anchor": "w", "justify": "left"},
        "self": {"bd": 1, "relief": tk.SOLID},
    }

    def __init__(
        self,
        master,
        checkbutton_names: list[str],
        pady: int = 0,
        checkbutton_kwargs: dict = None,
        **kwargs
    ):
        """Initialises CheckbuttonPanel.
        Merges user passed kwargs with default fallbacks.
        Creates and lays out all widgets within self.

        Args:
            master (_type_): Root window for self.
            checkbutton_names (list[str]): List of names for checkbuttons.
            pady (int, optional): Vertical pad between checkbuttons. Defaults to 0.
            checkbutton_kwargs (dict, optional): kwargs to use for checkbutton creation. Defaults to None.
        """

        # merge user provided self kwargs with default kwargs and call super's __init__
        super().__init__(
            master, **self._apply_fallback_kwargs(kwargs, self.DEFAULT_KWARGS["self"])
        )

        # store pad_y and checbutton_names directly as attributes
        self.checkbutton_names = checkbutton_names
        self.pady = pady

        # create list of boolean vars representing checkbutton states.
        self.checkbutton_vars = [
            tk.BooleanVar(value=self.DEFAULT_CHECKBUTTON_STATE)
            for _ in checkbutton_names
        ]

        # merge user provided and default kwargs for checkbutton creation.
        self.checkbutton_kwargs = self._apply_fallback_kwargs(
            checkbutton_kwargs, self.DEFAULT_KWARGS["checkbutton"]
        )

        # create and layout widgets.
        self._create_widgets()
        self._layout_widgets()

    @classmethod
    def _apply_fallback_kwargs(cls, kwargs: dict, default_kwargs: dict) -> dict:
        """Merge kwargs and default_kwargs dicts. kwargs
        overwrites default_kwargs if shared keys exist.

        Args:
            kwargs (dict): User-specified kwargs
            default_kwargs (dict): Default kwargs

        Returns:
            dict_: Merged kwargs
        """
        merged_kwargs = {
            **default_kwargs,
            **(kwargs or {}),
        }
        return merged_kwargs

    def _create_widgets(self):
        """Create all widgets for self"""

        # create list of checkbuttons using name and bool var attributes
        self.checkbuttons = [
            tk.Checkbutton(self, text=name, variable=var, **self.checkbutton_kwargs)
            for name, var in zip(self.checkbutton_names, self.checkbutton_vars)
        ]

    def _layout_widgets(self):
        """Lays out all widgets within self"""

        # pack checkbuttons vertically
        for cb in self.checkbuttons:
            cb.pack(side="top", fill="x", anchor="w", pady=(0, self.pady))

    def get_selected_items(self) -> list[str]:
        """Returns the names associated with all
        currently ticked checkbuttons.

        Returns:
            list[str]: Names of ticked checkbuttons
        """
        selected_items = [
            cb.cget("text")
            for cb, var in zip(self.checkbuttons, self.checkbutton_vars)
            if var.get() == 1
        ]
        return selected_items
