import os
import re


class StrUtils:

    @staticmethod
    def print_break(length: int):
        print("".join("-" for _ in range(length)))

    @staticmethod
    def extract_coil(file_path: str):
        if isinstance(file_path, list):
            file_path = file_path[0]
        file_path = os.path.basename(file_path)
        return file_path[: file_path.find("_")]

    @staticmethod
    def extract_orientation(file_path: str):
        if isinstance(file_path, list):
            file_path = file_path[0]
        file_path = os.path.basename(file_path)
        underscore_locs = [
            match.start() for match in re.finditer(re.escape("_"), file_path)
        ]
        slice_start = underscore_locs[0] + 1
        slice_end = None if len(underscore_locs) == 1 else underscore_locs[1]
        return file_path[slice_start:slice_end]
