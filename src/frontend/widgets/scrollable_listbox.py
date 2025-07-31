"""
scrollable_listbox.py

Script defining ScrollableListbox tk.Frame subclass. This class links a listbox
and a scrollbar together into one cohesive widget.

Written by Nathan Crossley, 2025.

"""

import tkinter as tk

from src.frontend.settings import FONT_TEXT


class ScrollableListbox(tk.Frame):
    """tk.Frame subclass representing a scrollable listbox.
    This consists of a listbox linked with a scrollbar.
    """

    # default kwargs for all used tk widgets. User can override.
    DEFAULT_KWARGS = {
        "listbox": {
            "selectmode": tk.EXTENDED,
            "font": FONT_TEXT,
            "bd": 1,
            "relief": tk.SOLID,
        },
        "scrollbar": {"orient": "vertical"},
    }

    def __init__(
        self,
        master,
        listbox_kwargs: dict = None,
        scrollbar_kwargs: dict = None,
        padx: int = 0,
        **kwargs
    ):
        """Initialises ScrollableListbox.
        Merges user passed kwargs with default fallbacks.
        Creates and lays out all widgets within self.

        Args:
            master (_type_): Root window for self.
            listbox_kwargs (dict, optional): kwargs to use for listbox creation. Defaults to None.
            scrollbar_kwargs (dict, optional): kwargs to use for scrollbar creation. Defaults to None.
            padx (int, optional): padding between listbox and scrollbar. Defaults to 0.
        """
        super().__init__(master, **kwargs)

        # store padx as attribute
        self.padx = padx

        # get widget kwargs by merging user provided kwargs with defaults
        self.listbox_kwargs = self._apply_fallback_kwargs(
            listbox_kwargs, self.DEFAULT_KWARGS["listbox"]
        )
        self.scrollbar_kwargs = self._apply_fallback_kwargs(
            scrollbar_kwargs, self.DEFAULT_KWARGS["scrollbar"]
        )

        # create and layout widgets
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
        """Creates all widgets for self"""

        # create listbox and scrollbar
        self.listbox = tk.Listbox(self, **self.listbox_kwargs)
        self.scrollbar = tk.Scrollbar(self, **self.scrollbar_kwargs)

        # link listbox and scrollbar
        self.listbox.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.listbox.yview)

    def _layout_widgets(self):
        """Layout all widgets within self"""

        # pack listbox and scrollbar side by side with padx separation
        self.listbox.pack(side="left", fill="both", expand=True, padx=(0, self.padx))
        self.scrollbar.pack(side="right", fill="y")

    def populate(self, items: list[str]):
        """Clears listbox and populates it with
        a list of items.

        Args:
            items (list[str]): Items to populate listbox with.
        """
        self.listbox.delete(0, tk.END)
        for item in items:
            self.listbox.insert(tk.END, item)

    def get_selected_items(self) -> list[str]:
        """Returns items currently selected in
        listbox.

        Returns:
            list[str]: List of selected items.
        """

        selection = [self.listbox.get(i) for i in self.listbox.curselection()]
        return selection
