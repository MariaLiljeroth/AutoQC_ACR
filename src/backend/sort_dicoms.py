from pathlib import Path
import pydicom

from shared.queueing import get_queue
from backend.utils import quick_check_dicom


class DicomSorter:
    """Class representing the DICOM sorting process."""

    def __init__(self, dir: Path):
        self.dir = dir
        self.uid_suffix_mapper = {}
        self.tracked_sDescrips = []

    def run(self):
        """Entry function for running DICOM sorting process"""
        self.dcms = self.get_valid_DICOMs()
        for dcm in self.dcms:
            # get UID and SeriesDescription tags and populate trackers.
            metadata = pydicom.dcmread(dcm)
            uid, sDescrip = metadata.SeriesInstanceUID, metadata.SeriesDescription
            self.populate_uid_suffix_mapper(uid, sDescrip)

            # Work out target folder path, create and move DICOM there.
            target_folder = (
                self.dir / sDescrip / self.uid_suffix_mapper[uid]
                if hasattr(metadata, "PixelData")
                else self.dir / "NoImageData"
            )
            target_folder.mkdir(exist_ok=True)
            dcm.rename(target_folder / dcm.name)
            get_queue().put(
                ("PROGRESS_BAR_UPDATE", "DICOM_SORTING", 1 / len(self.dcms) * 100)
            )
        get_queue().put(("TASK_COMPLETE", "DICOM_SORTING"))

    def get_valid_DICOMs(self) -> list[pydicom.FileDataset]:
        """Returns a list of all DICOM files within the parent folder.

        Returns:
            list[pydicom.FileDataSet]: List of DICOM files.
        """
        files = list(self.dir.glob("*"))
        dcms = []
        for f in files:
            if quick_check_dicom(f):
                dcms.append(f)
            get_queue().put(
                ("PROGRESS_BAR_UPDATE", "DICOM_CHECKING", 1 / len(files) * 100)
            )
        get_queue().put(("TASK_COMPLETE", "DICOM_CHECKING"))
        return dcms

    def populate_uid_suffix_mapper(self, uid: str, sDescrip: str):
        """Populates the dict self.uid_suffix_mapper if necessary. This dict
        maps UID values to different suffixes. This allows duplicate data sets
        for Siemens scanners to be sorted into a subdirectory within the main
        directory.

        Args:
            uid (str): SeriesInstanceUID DICOM tag
            sDescrip (str): SeriesDescription DICOM tag
        """
        if sDescrip not in self.tracked_sDescrips:
            self.tracked_sDescrips.append(sDescrip)
            self.uid_suffix_mapper[uid] = ""
        else:
            if uid not in self.uid_suffix_mapper:
                self.uid_suffix_mapper[uid] = "helper_data_set"
