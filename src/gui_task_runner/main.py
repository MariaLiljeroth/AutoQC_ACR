import sys
from pathlib import Path

sys.path.append(
    str(
        Path(__file__).parent.parent.resolve()
        / "Scottish-Medium-ACR-Analysis-Framework"
    )
)

from config_window import ConfigWindow

if __name__ == "__main__":
    config_window = ConfigWindow()
    config_window.mainloop()
