from pathlib import Path
import pydicom
from pydicom.errors import InvalidDicomError

from shared.queueing import get_queue
from backend.utils import nested_dict
from backend.dcm_objects.dcm_subdirectory import DcmSubdirectory
from backend.dcm_objects.dcm_extended import DcmExtended


class DcmSorter:
    """Class representing the DICOM sorting process.

    Instance attributes:
        dir (Path): Directory containing DICOM files.
        uid_suffix_mapper (dict[str, str]): Dictionary mapping SeriesInstanceUID to suffixes.
            Useful for Siemens scanners with duplicate data sets (see SNR by subtraction)
        tracked_sDescrips (list[str]): List of SeriesDescription values that have been tracked.
    """

    SERIES_DESC_CLEANER = str.maketrans({c: "_" for c in '\\/:*?"<>|'})

    def __init__(self, dir: Path):
        """Initialises the DicomSorter class.

        Args:
            dir (Path): Directory containing DICOM files to sort.
        """
        self.dir = dir

    def run(self):
        """Entry function for running DICOM sorting process"""
        self.dcms = self.read_loose_dcms()

        self.series_desc_counter = nested_dict()
        self.uid_suffix_mapper = nested_dict()

        self.sorted_subdirs = {}

        for dcm in self.dcms:
            dcm = self.modify_series_desc(dcm)

            # Send update to queue for progress bar visuals.
            get_queue().put(
                ("PROGRESS_BAR_UPDATE", "DICOM_SORTING", 1 / len(self.dcms) * 100)
            )
        get_queue().put(("TASK_COMPLETE", "DICOM_SORTING", self.sorted_subdirs))

    def sort_dcms_into_subdirs(self) -> list[Path]:
        glob_paths = list([f for f in self.dir.glob("*")])
        file_paths = [f for f in glob_paths if f.is_file()]
        dir_paths = [d for d in glob_paths if d.is_dir()]

        sorted_subdirs = {}

        for fp in file_paths:
            try:
                dcm = DcmExtended(fp)
                target_subdir = (
                    self.dir / dcm.metadata.SeriesDescription
                    if hasattr(dcm.metadata, "PixelData")
                    else "no_image_data"
                )
                if target_subdir not in sorted_subdirs:
                    sorted_subdirs[target_subdir] = DcmSubdirectory(target_subdir)

                sorted_subdirs[target_subdir].receive_dcm(dcm)

            except InvalidDicomError:
                pass

        for dp in dir_paths:
            file_paths = list([f for f in dp.glob("*")])
            for fp in file_paths:
                try:
                    dcm = DcmExtended(fp)
                    if dp not in sorted_subdirs:
                        sorted_subdirs[target_subdir] = DcmSubdirectory(dp)

                    sorted_subdirs[target_subdir].receive_dcm(dcm)
                except InvalidDicomError:
                    pass

            get_queue().put(
                ("PROGRESS_BAR_UPDATE", "DICOM_READING", 1 / len(file_paths) * 100)
            )
        get_queue().put(("TASK_COMPLETE", "DICOM_READING"))
        return dcms

    def modify_series_desc(self, dcm):
        uid = (dcm.metadata.SeriesInstanceUID,)
        series_desc = dcm.metadata.SeriesDescription

        series_desc_safe = series_desc.translate(self.SERIES_DESC_CLEANER)
        if series_desc_safe not in self.series_desc_counter:
            self.series_desc_counter[series_desc_safe] = 1
            self.uid_suffix_mapper[uid] = ""

        elif uid not in self.uid_suffix_mapper:
            self.uid_suffix_mapper[uid] = (
                f"_{self.series_desc_counter[series_desc_safe]}"
            )

        dcm.metadata.SeriesDescription = (
            f"{series_desc_safe}{self.uid_suffix_mapper[uid]}"
        )

        return dcm
