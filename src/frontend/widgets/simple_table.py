import tkinter as tk
import pandas as pd

from src.frontend.settings import FONT_TEXT


class SimpleTable(tk.Frame):

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
        super().__init__(master, **kwargs)
        self.entry_kwargs = self._apply_fallback_kwargs(
            entry_kwargs, self.DEFAULT_KWARGS["entry"]
        )
        self.label_kwargs = self._apply_fallback_kwargs(
            label_kwargs, self.DEFAULT_KWARGS["label"]
        )
        self.row_headers = row_headers
        self.col_headers = col_headers
        self.border_pad = border_pad
        self.force_numerical = force_numerical

        self._create_widgets()
        self._layout_widgets()

    @classmethod
    def _apply_fallback_kwargs(cls, kwargs, default_kwargs):
        safe_kwargs = {
            **default_kwargs,
            **(kwargs or {}),
        }
        return safe_kwargs

    def _create_widgets(self):
        self._create_frames()
        self._create_entry_dataframe()
        self._create_labels()

    def _create_frames(self):
        self.frame_inner = tk.Frame(self)

    def _create_entry_dataframe(self):
        vcmd = (self.register(self._all_chars_allowed), "%P")
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
        self.entry_df = pd.DataFrame(
            entries,
            index=self.row_headers,
            columns=self.col_headers,
        )

    def _create_labels(self):
        self.labels_rows = [
            tk.Label(self.frame_inner, text=header, justify="left", **self.label_kwargs)
            for header in self.row_headers
        ]
        self.labels_cols = [
            tk.Label(self.frame_inner, text=header, **self.label_kwargs)
            for header in self.col_headers
        ]

    def _layout_widgets(self):
        self._layout_frames()
        self._layout_entries()
        self._layout_labels()

    def _layout_frames(self):
        self.frame_inner.pack(padx=self.border_pad, pady=self.border_pad)

    def _layout_entries(self):
        for i in range(self.entry_df.shape[0]):
            for j in range(self.entry_df.shape[1]):
                entry = self.entry_df.iloc[i, j]
                entry.grid(row=i + 1, column=j + 1)

    def _layout_labels(self):
        for i, label in enumerate(self.labels_rows):
            label.grid(row=i + 1, column=0, sticky="w", padx=(0, 5))

        for i, label in enumerate(self.labels_cols):
            label.grid(row=0, column=i + 1)

    @staticmethod
    def _all_chars_allowed(s: str) -> bool:
        if s == "":
            return True
        # Allow only one minus at start and one decimal point
        if s.count("-") > 1 or s.count(".") > 1:
            return False
        if "-" in s and s.index("-") != 0:
            return False
        allowed = set("0123456789-.")
        return all(c in allowed for c in s)

    def get_as_pandas_df(self):
        vals = self.entry_df.map(lambda entry: entry.get())
        return vals


pass
