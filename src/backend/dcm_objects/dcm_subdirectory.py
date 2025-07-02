from pathlib import Path
from pydicom.dataset import FileDataset
from datetime import datetime


class DcmSubdirectory:

    def __init__(self, path: Path):
        path.mkdir(exist_ok=True)
        self.path = path
        self.dcms = []
        self.datetime = None

    def receive_dcm(self, dcm):
        # Move dicom file location
        new_path = self.path / dcm.path.name
        dcm.path.rename(new_path)
        dcm.path = new_path

        # Add dcm to self.dcms
        self.dcms.append(dcm)

        # Get series datetime object and initialise instance attribute
        dcm_date = dcm.metadata.get("SeriesDate")
        dcm_time = dcm.metadata.get("SeriesTime")

        if dcm_date and dcm_time and self.datetime is None:
            dcm_date_obj = datetime.strptime(dcm_date, "%Y%m%d").date()
            dcm_time_obj = datetime.strptime(
                dcm_time, f"%H%M%S{'.%f' if '.' in dcm_time else ''}"
            ).time()
            dcm_datetime_obj = datetime.combine(dcm_date_obj, dcm_time_obj)
            if self.datetime is None:
                self.datetime = dcm_datetime_obj
