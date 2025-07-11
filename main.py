"""
main.py

This script is the entry point for AutoQC_ACR.
Controls most toplevel behaviour, creating App class and managing pyinstaller splash screen.

Written by Nathan Crossley 2025

"""

import sys
import multiprocessing as mp
import matplotlib

matplotlib.use("Agg")

from src.frontend.app import App

# bool determining whether script is run from .exe (script frozen) or in IDE.
frozen = getattr(sys, "frozen", False)

if __name__ == "__main__":
    mp.freeze_support()

    # create instance of App class.
    app = App()
    if frozen:
        # if script frozen, close splash screen after App initialised.
        import pyi_splash  # type: ignore

        pyi_splash.close()

    # run app mainloop
    app.mainloop()
