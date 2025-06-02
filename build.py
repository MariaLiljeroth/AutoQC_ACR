import subprocess
import sys
from pathlib import Path

version = input("Please enter version number for new build: ").strip()

if not version:
    print("No version number entered, aborting build process.")
    sys.exit()

subprocess.run(
    [
        "pyinstaller",
        "--noconfirm",
        "--clean",
        "--distpath",
        f"dist/AutoQC_ACR_v{version}",
        "AutoQC_ACR.spec",
    ]
)
