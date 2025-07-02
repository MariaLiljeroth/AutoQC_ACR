"""
main.py

This script is the entry point for AutoQC_ACR.
Controls most toplevel behaviour, creating App class and managing pyinstaller splash screen.

"""

import sys
import multiprocessing as mp
import matplotlib

matplotlib.use("Agg")

from frontend.app import App

frozen = getattr(sys, "frozen", False)

if __name__ == "__main__":
    mp.freeze_support()
    app = App()
    if frozen:
        import pyi_splash  # type: ignore

        pyi_splash.close()
    app.mainloop()
