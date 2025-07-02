from pathlib import Path
import pydicom


class DcmExtended:

    def __init__(self, path: Path):
        self.metadata = pydicom.dcmread(path)
        self.path = path
