# This script is the entry point for AutoQC_ACR. An instance of the frontend App class is created and run

import sys
from pathlib import Path
import multiprocessing as mp

# Append hazenlib parent directory to sys.path as required for imports internal to Scottish-Medium-ACR-Analysis-Framework
frozen = getattr(sys, "frozen", False)
main_path = sys.argv[0] if frozen else __file__
sys.path.append(str((Path(main_path).parent / "backend/smaaf").resolve()))

from frontend.app import App

if __name__ == "__main__":
    mp.freeze_support()
    app = App()
    if frozen:
        import pyi_splash  # type: ignore

        pyi_splash.close()
    app.mainloop()
