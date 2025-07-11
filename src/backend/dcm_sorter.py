"""
dcm_sorter.py

This script defines the DcmSorter class which is used to check for and sort a directory of unsorted dicoms
into separate folders. Separation is done based on dicom tags, namely the series description and series instance uid.

Written by Nathan Crossley 2025

"""

from pathlib import Path
import pydicom

from src.shared.queueing import get_queue
from src.backend.utils import quick_check_dicom


class DcmSorter:
    """Class to sort a directory of unsorted dicoms into separate folders
    based on their dicom tags (namely series description and series instance uid).

    Instance attributes:
        dir (Path): Directory containing DICOM files to sort.
        uid_suffix_mapper (dict[str, str]): Dictionary mapping SeriesInstanceUID tag to suffixes
            for sorted folders. Useful for Siemens scanners (SNR by subtraction)
        tracked_series_descs (list[str]): List of SeriesDescription tag values that have been encountered
            during the sorting process.
    """

    # translation table to replace incompatible characters in file paths with underscores.
    TRANSLATION_TABLE = str.maketrans({c: "_" for c in '\\/:*?"<>|'})

    def __init__(self, dir: Path):
        """Initialises the DicomSorter class by setting up instance attributes.

        Args:
            dir (Path): Directory containing DICOM files to sort.
        """

        # store reference to dicom directory path
        self.dir = dir

        # set up mappers and trackers for dicom sorting process
        self.uid_suffix_mapper = {}
        self.tracked_series_descs = []

    def run(self):
        """Entry function for running DICOM sorting process"""

        # get a list of valid dicom files from dicom directory
        self.dcms = self.get_valid_DICOMs()

        # process each dicom individually
        for dcm in self.dcms:

            # read dicom metadata
            metadata = pydicom.dcmread(dcm)

            # These three tags needed for meaningful analysis. Store presence as bool
            required_tags = ("SeriesInstanceUID", "SeriesDescription", "PixelData")
            required_tags_exist = all([hasattr(metadata, tag) for tag in required_tags])

            # Process dicoms that may be used (those with expected tags)
            if required_tags_exist:

                # get uid and series description tags
                uid = metadata.get("SeriesInstanceUID")
                series_desc = metadata.get("SeriesDescription")

                # clean series description tag with translation table to prevent file path errors and bugs
                series_desc = series_desc.translate(self.TRANSLATION_TABLE)

                # set up suffix that corresponds to specific uid
                self.populate_uid_suffix_mapper(uid, series_desc)

                # construct folder path to move dicom to using series description and assigned suffix
                target_folder = self.dir / series_desc / self.uid_suffix_mapper[uid]

            else:
                # for invalid dicoms, move to miscellaneous folder as clean solution
                target_folder = self.dir / "dcm_tags_missing"

            # make target folder in os (folder that dcm should be moved to)
            target_folder.mkdir(exist_ok=True)

            # move specific dcm to folder
            dcm.rename(target_folder / dcm.name)

            # Send signal to queue to indicate that a singular dcm has been sorted
            get_queue().put(
                ("PROGRESS_BAR_UPDATE", "DICOM_SORTING", 1 / len(self.dcms) * 100)
            )

        # send signal that dcm sorting process is complete
        get_queue().put(("TASK_COMPLETE", "DICOM_SORTING"))

    def get_valid_DICOMs(self) -> list[Path]:
        """Returns a list of paths to all DICOM files within parent dicom directory

        Returns:
            list[Path]: List of paths to DICOM files within dicom directory.
        """

        # get list of all files (non-recursive) and initialise list for dcm storage
        files = list(self.dir.glob("*"))
        dcm_paths = []

        for f in files:

            # if file is a dcm, append path to list
            if quick_check_dicom(f):
                dcm_paths.append(f)

            # send signal to update progress bar to say that one file has been checked
            get_queue().put(
                ("PROGRESS_BAR_UPDATE", "DICOM_CHECKING", 1 / len(files) * 100)
            )

        # send signal to queue to indicate that dicom checking process has been completed
        get_queue().put(("TASK_COMPLETE", "DICOM_CHECKING"))

        return dcm_paths

    def populate_uid_suffix_mapper(self, uid: str, series_desc: str):
        """Populates the dict self.uid_suffix_mapper with a suffix corresponding
        to that particular uid.

        Args:
            uid (str): SeriesInstanceUID DICOM tag for specific dcm
            series_desc (str): SeriesDescription DICOM tag for specific dcm
        """

        # if series description not tracked yet, append to series description tracker
        # and update suffix mapper with a blank suffix
        if series_desc not in self.tracked_series_descs:
            self.tracked_series_descs.append(series_desc)
            self.uid_suffix_mapper[uid] = ""

        # otherwise add a "helper_data_set" suffix associated with that uid
        else:
            if uid not in self.uid_suffix_mapper:
                self.uid_suffix_mapper[uid] = "helper_data_set"
