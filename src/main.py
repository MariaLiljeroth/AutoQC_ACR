# This script is the entry point for AutoQC_ACR. An instance of the frontend App class is created and run

import sys
from pathlib import Path

# Append hazenlib parent directory to sys.path as required for imports internal to Scottish-Medium-ACR-Analysis-Framework
sys.path.append(str((Path(__file__).parent / "backend/smaaf").resolve()))
from frontend.app import App

if __name__ == "__main__":
    app = App()
    app.mainloop()
