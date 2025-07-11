"""build.py

A build script for AutoQC_ACR, which bundles the project into a distributable directory using pyinstaller.
This script pulls the pysintaller .spec file, which details the specifics of the bundling process.
When the script is run, the user is prompted to enter a version number for the build. The suggested version number convention
is MAJOR.MINOR.REVISION.(BUILD), as is standard. The bundled project will be available in dist/

"""

import subprocess
import sys


# Specify version number for build - MAJOR.MINOR.REVISION(.BUILD)
version = input("Please enter version number for new build: ").strip()

# Do not build if no version number entered.
if not version:
    print("No version number entered, aborting build process.")
    sys.exit()


# Determine command line interface args and pass to subprocess.
# Can insert --clean flag into cli_args for totally fresh build (no cache dependency).
cli_args = [
    "pyinstaller",
    "--noconfirm",
    "--distpath",
    f"dist/AutoQC_ACR_v{version}",
    "AutoQC_ACR.spec",
]
subprocess.run(cli_args)
