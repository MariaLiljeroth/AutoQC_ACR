import tkinter as tk


def populate_entry_widget(entry_widget: tk.Entry, text: str):
    """Populates an entry widget with a text string, deleting current content.

    Args:
        entry_widget (tk.Entry): Entry widget to populate.
        text (str): String to populate entry widget with.
    """
    # delete all content in entry widget
    entry_widget.delete(0, tk.END)

    # insert text within entry widget
    entry_widget.insert(0, text)

    # ensure that start of entry widget's text is visible
    entry_widget.icursor(tk.END)
    entry_widget.xview_moveto(1)
