import tkinter as tk

from src.frontend.settings import FONT_TEXT


class ScrollableListbox(tk.Frame):
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
        super().__init__(master, **kwargs)
        self.padx = padx
        self.listbox_kwargs = self._apply_fallback_kwargs(
            listbox_kwargs, self.DEFAULT_KWARGS["listbox"]
        )
        self.scrollbar_kwargs = self._apply_fallback_kwargs(
            scrollbar_kwargs, self.DEFAULT_KWARGS["scrollbar"]
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
        self.listbox = tk.Listbox(self, **self.listbox_kwargs)
        self.scrollbar = tk.Scrollbar(self, **self.scrollbar_kwargs)

        self.listbox.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.listbox.yview)

    def _layout_widgets(self):
        self.listbox.pack(side="left", fill="both", expand=True, padx=(0, self.padx))
        self.scrollbar.pack(side="right", fill="y")

    def populate(self, items: list):
        self.listbox.delete(0, tk.END)
        for item in items:
            self.listbox.insert(tk.END, item)

    def get_selected_items(self):
        selection = [self.listbox.get(i) for i in self.listbox.curselection()]
        return selection
