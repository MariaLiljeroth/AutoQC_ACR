import tkinter as tk

root = tk.Tk()
root.title("Dropdown Example")

options = ["Option 1", "Option 2", "Option 3"]
selected = tk.StringVar(value=options[0])  # default

dropdown = tk.OptionMenu(root, selected, *options)
dropdown.pack(padx=10, pady=10)


def show_selection():
    print("You selected:", selected.get())


tk.Button(root, text="Submit", command=show_selection).pack()

root.mainloop()
