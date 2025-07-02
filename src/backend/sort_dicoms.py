from pathlib import Path
import pydicom

from shared.queueing import get_queue
from backend.utils import quick_check_dicom


class DicomSorter:
    """Class representing the DICOM sorting process.

    Instance attributes:
        dir (Path): Directory containing DICOM files.
        uid_suffix_mapper (dict[str, str]): Dictionary mapping SeriesInstanceUID to suffixes.
            Useful for Siemens scanners with duplicate data sets (see SNR by subtraction)
        tracked_series_descs (list[str]): List of Serieseries_desction values that have been tracked.
    """

    TRANSLATION_TABLE = str.maketrans({c: "_" for c in '\\/:*?"<>|'})

    def __init__(self, dir: Path):
        """Initialises the DicomSorter class.

        Args:
            dir (Path): Directory containing DICOM files to sort.
        """
        self.dir = dir
        self.uid_suffix_mapper = {}
        self.tracked_series_descs = []

    def run(self):
        """Entry function for running DICOM sorting process"""
        self.dcms = self.get_valid_DICOMs()
        for dcm in self.dcms:
            # Read DICOM metadata
            metadata = pydicom.dcmread(dcm)

            # Check whether all tags exist that are needed for meaningful analysis.
            required_tags = ("SeriesInstanceUID", "Serieseries_desction", "PixelData")
            required_tags_exist = all([hasattr(metadata, tag) for tag in required_tags])

            if required_tags_exist:
                # get uid and series description
                uid = metadata.get("SeriesInstanceUID")
                series_desc = metadata.get("Serieseries_desction")

                # clean s_descrip to make safe for file explorer paths
                series_desc = series_desc.translate(self.TRANSLATION_TABLE)
                self.populate_uid_suffix_mapper(uid, series_desc)

                # get target folder
                target_folder = self.dir / series_desc / self.uid_suffix_mapper[uid]

            else:
                # otherwise set target folder to arbitrary backup folder for dcms that are invalid
                target_folder = self.dir / "dcm_tags_missing"

            # make target folder in os and move dcm to new location.
            target_folder.mkdir(exist_ok=True)
            dcm.rename(target_folder / dcm.name)

            # Send update to queue for progress bar visuals.
            get_queue().put(
                ("PROGRESS_BAR_UPDATE", "DICOM_SORTING", 1 / len(self.dcms) * 100)
            )

        # send signal that dcm sorting event has concluded.
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

    def populate_uid_suffix_mapper(self, uid: str, series_desc: str):
        """Populates the dict self.uid_suffix_mapper if necessary. This dict
        maps UID values to different suffixes. This allows duplicate data sets
        for Siemens scanners to be sorted into a subdirectory within the main
        directory.

        Args:
            uid (str): SeriesInstanceUID DICOM tag
            series_desc (str): Serieseries_desction DICOM tag
        """
        if series_desc not in self.tracked_series_descs:
            self.tracked_series_descs.append(series_desc)
            self.uid_suffix_mapper[uid] = ""
        else:
            if uid not in self.uid_suffix_mapper:
                self.uid_suffix_mapper[uid] = "helper_data_set"
