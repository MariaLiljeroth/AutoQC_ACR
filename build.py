import subprocess
import sys

version = input("Please enter version number for new build: ").strip()

if not version:
    print("No version number entered, aborting build process.")
    sys.exit()

# Can insert --clean flag into cli_args for totally fresh build
cli_args = [
    "pyinstaller",
    "--noconfirm",
    "--distpath",
    f"dist/AutoQC_ACR_v{version}",
    "AutoQC_ACR.spec",
]

subprocess.run(cli_args)
