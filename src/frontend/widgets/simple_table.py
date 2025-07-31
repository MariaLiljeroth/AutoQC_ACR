"""
simple_table.py

Script defines SimpleTable, a subclass of tk.Frame. This class is used to represent
a simple table widget which is a combination of entries and labels. The user can type
into the entries and these values can be returned.

Written by Nathan Crossley, 2025.
"""

import tkinter as tk
import pandas as pd

from src.frontend.settings import FONT_TEXT


class SimpleTable(tk.Frame):
    """Subclass of tk.Frame used to define a
    simple table widget. Consists of a collection of
    entries and labels. User can type into the entries.
    """

    # default kwargs for all used tk widgets. User can override.
    DEFAULT_KWARGS = {
        "entry": {"bd": 1, "relief": tk.SOLID, "font": FONT_TEXT},
        "label": {"font": FONT_TEXT},
    }

    def __init__(
        self,
        master,
        row_headers: list[str],
        col_headers: list[str],
        entry_kwargs: dict = None,
        label_kwargs: dict = None,
        border_pad: int = 0,
        force_numerical: bool = True,
        **kwargs,
    ):
        """Initialises SimpleTable.
        Merges user passed kwargs with default fallbacks.
        Creates and lays out all widgets within self.

        Args:
            master (_type_): Root window for self
            row_headers (list[str]): List of row headers for table.
            col_headers (list[str]): List of column headers for table.
            entry_kwargs (dict, optional): kwargs to use for entry creation. Defaults to None.
            label_kwargs (dict, optional): kwargs to use for label creation. Defaults to None.
            border_pad (int, optional): Padding between "table" and edge of frame. Defaults to 0.
            force_numerical (bool, optional): Bool to only allow the user to enter numerical data
                into the table. Defaults to True.
        """
        super().__init__(master, **kwargs)

        # get widget kwargs by merging user provided kwargs with defaults
        self.entry_kwargs = self._apply_fallback_kwargs(
            entry_kwargs, self.DEFAULT_KWARGS["entry"]
        )
        self.label_kwargs = self._apply_fallback_kwargs(
            label_kwargs, self.DEFAULT_KWARGS["label"]
        )

        # store other args as attributes.
        self.row_headers = row_headers
        self.col_headers = col_headers
        self.border_pad = border_pad
        self.force_numerical = force_numerical

        # create and layout all widgets.
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
        """Creates all widgets for self. Individual
        functions provided for individual widget types."""
        self._create_frames()
        self._create_entry_dataframe()
        self._create_labels()

    def _create_frames(self):
        """Create all frames for self"""

        # inner frame creation so can pad table and edge of self.
        self.frame_inner = tk.Frame(self)

    def _create_entry_dataframe(self):
        """Creates a pandas dataframe of entry widgets so can be accessed
        by column and row headers.
        """

        # validate command to only accept numerical characters in entries
        vcmd = (self.register(self._all_chars_numerical), "%P")

        # create 2d grid of entry widgets
        entries = [
            [
                tk.Entry(
                    self.frame_inner,
                    **self.entry_kwargs,
                    validate="key" if self.force_numerical else None,
                    validatecommand=vcmd if self.force_numerical else None,
                )
                for _ in self.col_headers
            ]
            for _ in self.row_headers
        ]

        # create pandas dataframe from grid of entry widgets and col/row headers.
        self.entry_df = pd.DataFrame(
            entries,
            index=self.row_headers,
            columns=self.col_headers,
        )

    def _create_labels(self):
        """Creates all labels for self."""

        # list of labels for row headers
        self.labels_rows = [
            tk.Label(self.frame_inner, text=header, justify="left", **self.label_kwargs)
            for header in self.row_headers
        ]

        # list of labels for col headers
        self.labels_cols = [
            tk.Label(self.frame_inner, text=header, **self.label_kwargs)
            for header in self.col_headers
        ]

    def _layout_widgets(self):
        """Lays out all widgets. Individual functions provided
        for individual widget types."""
        self._layout_frames()
        self._layout_entries()
        self._layout_labels()

    def _layout_frames(self):
        """Lays out all frames for self"""
        self.frame_inner.pack(padx=self.border_pad, pady=self.border_pad)

    def _layout_entries(self):
        """Lays of all entries within entry dataframe"""
        for i in range(self.entry_df.shape[0]):
            for j in range(self.entry_df.shape[1]):
                entry = self.entry_df.iloc[i, j]
                entry.grid(row=i + 1, column=j + 1)

    def _layout_labels(self):
        """Lays out all labels for self"""

        # layout row labels
        for i, label in enumerate(self.labels_rows):
            label.grid(row=i + 1, column=0, sticky="w", padx=(0, 5))

        # layout column labels
        for i, label in enumerate(self.labels_cols):
            label.grid(row=0, column=i + 1)

    @staticmethod
    def _all_chars_numerical(s: str) -> bool:
        """Function to test whether all characters in a string
        are numerical.

        Args:
            s (str): String to test.

        Returns:
            bool: True if all characters are numerical, False otherwise.
        """

        # allow empty string so entry can be blank.
        if s == "":
            return True

        # allow only one minus at start and one decimal point
        if s.count("-") > 1 or s.count(".") > 1:
            return False
        if "-" in s and s.index("-") != 0:
            return False

        # get set of allowed numerical values.
        allowed = set("0123456789-.")

        # check to see if all chars in string are in set.
        return all(c in allowed for c in s)

    def get_current_state(self) -> pd.DataFrame:
        """Returns the current state of table as a pandas dataframe
        by getting all entry values.

        Returns:
            pd.DataFrame: Current state of table.
        """

        # map get func across entry dataframe to get dataframe of current entry values.
        vals = self.entry_df.map(lambda entry: entry.get())

        return vals
