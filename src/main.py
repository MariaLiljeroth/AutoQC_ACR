# This script is the entry point for AutoQC_ACR. An instance of the frontend App class is created and run

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
