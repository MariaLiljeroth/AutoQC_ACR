import sys
from pathlib import Path
from tkinter import filedialog
import pydicom

"""DICOM DIRECTORY SELECTION"""

msg_intro = """
------------------------------------------------------------------
The SeriesInstanceUID dcm tag is unique to a particular MR series.
This tag can hence be utilised to select all dcms from a particular series and modify any incorrect tags.

In the case of AutoQC_ACR, it can be the case where the SeriesDescription of particular dcms are incorrect.
This script can rectify this issue by assigning a user-provided value to the SeriesDescription tag.

First, please select the unsorted dcm directory (full of unsorted dcms)..."""

print(msg_intro)
dcm_dir = filedialog.askdirectory(title="Select Dcm Directory...")

if not dcm_dir:
    print("Dcm Directory was not selected. Quitting...")
    sys.exit()

dcm_dir = Path(dcm_dir)
msg_dir_selected = f"""
Directory selected: {dcm_dir.resolve()}
--------------------{''.join(['-' for _ in range(len(str(dcm_dir.resolve())))])}"""
print(msg_dir_selected)

"""USER INPUTS"""

msg_get_uid = """
Next, please input the SeriesInstanceUID for the series that you want to modify the SeriesDescription for:

-> """

uid_find = input(msg_get_uid)

msg_get_series_descrip = """
Finally, what would you like the SeriesDescription tag to be modified to?

-> """

series_descrip_set = input(msg_get_series_descrip)

"""DCM MODIFICATION"""
for item in dcm_dir.iterdir():
    if item.is_file():
        try:
            metadata = pydicom.dcmread(item)
            if hasattr(metadata, "SeriesDescription") and hasattr(
                metadata, "SeriesInstanceUID"
            ):
                uid = metadata.SeriesInstanceUID
                if uid == uid_find:
                    metadata.SeriesDescription = series_descrip_set
                    metadata.save_as(item)
                    print(f"Series Description changed for file {item.name}")
        except:
            pass
