import sys
from pathlib import Path

sys.path.append(str((Path(__file__).parent / "backend/smaaf").resolve()))
from frontend.app import App

if __name__ == "__main__":
    app = App()
    app.mainloop()
