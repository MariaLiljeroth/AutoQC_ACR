import tkinter as tk

from src.frontend.settings import FONT_TEXT


class CheckbuttonPanel(tk.Frame):
    DEFAULT_CHECKBUTTON_STATE = True
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
        super().__init__(
            master, **self._apply_fallback_kwargs(kwargs, self.DEFAULT_KWARGS["self"])
        )
        self.checkbutton_names = checkbutton_names
        self.pady = pady
        self.checkbutton_vars = [
            tk.BooleanVar(value=self.DEFAULT_CHECKBUTTON_STATE)
            for _ in checkbutton_names
        ]
        self.checkbutton_kwargs = self._apply_fallback_kwargs(
            checkbutton_kwargs, self.DEFAULT_KWARGS["checkbutton"]
        )

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
        self.checkbuttons = [
            tk.Checkbutton(self, text=name, variable=var, **self.checkbutton_kwargs)
            for name, var in zip(self.checkbutton_names, self.checkbutton_vars)
        ]

    def _layout_widgets(self):
        for cb in self.checkbuttons:
            cb.pack(side="top", fill="x", anchor="w", pady=(0, self.pady))

    def get_selected_items(self):
        selected_items = [
            cb.cget("text")
            for cb, var in zip(self.checkbuttons, self.checkbutton_vars)
            if var.get() == 1
        ]
        return selected_items
