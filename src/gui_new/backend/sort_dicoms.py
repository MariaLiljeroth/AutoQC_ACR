from pathlib import Path
import pydicom

from shared.global_queue import get_queue
from backend.utils import quick_check_dicom


def sort_dicoms(dir: Path):
    files = list(dir.glob("*"))
    dcms = []
    for f in files:
        if quick_check_dicom(f):
            dcms.append(f)
        get_queue().put(("UPDATE: PROGRESS BAR DICOM CHECKING", 1 / len(files) * 100))
    get_queue().put("FINISH: PROGRESS BAR DICOM CHECKING")

    for dcm in dcms:
        target_folder = dir / pydicom.dcmread(dcm).SeriesDescription
        target_folder.mkdir(exist_ok=True)
        dcm.rename(target_folder / dcm.name)
        get_queue().put(("UPDATE: PROGRESS BAR DICOM SORTING", 1 / len(dcms) * 100))
    get_queue().put("FINISH: PROGRESS BAR DICOM SORTING")
