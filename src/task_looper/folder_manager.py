import sys
from pathlib import Path
import pydicom
import logging

import tkinter as tk
from tkinter import filedialog

import config
from utils import substring_matcher
from multi_dir_selector import MultiDirSelector


class FolderManager:

    def run(self):
        root = tk.Tk()
        root.attributes("-topmost", True)
        root.withdraw()

        logging.info(
            "Please select input data directory. DICOMs can be unsorted or previously auto-sorted."
        )
        self.in_dir = self.select_dir("Please select input data directory...")
        logging.info("Please select output data directory.")
        self.out_dir = self.select_dir("Please select output data directory...")
        root.destroy()

        self.sort_loose_dcms()
        in_subdirs = [x for x in self.in_dir.iterdir() if x.is_dir()]

        logging.info("Please select all subdirectories to process.")
        self.in_subdirs, self.to_archive = MultiDirSelector(in_subdirs).run()

        self.archive_dirs()
        self.enforce_naming_conventions()
        self.out_subdirs = self.create_out_subdirs()

    @staticmethod
    def select_dir(title: str) -> Path:
        dir = filedialog.askdirectory(title=title)
        if dir == "":
            logging.error("No directory selected. Exiting programme.")
            sys.exit()
        else:
            return Path(dir)

    def sort_loose_dcms(self):
        dcms = [
            x for x in self.in_dir.glob("*") if x.is_file() and pydicom.misc.is_dicom(x)
        ]
        if dcms:
            logging.info("Sorting loose DICOM files into folders.")
        else:
            logging.info("No loose DICOM files found. Sorting not required.")

        for dcm in dcms:
            target_folder = self.in_dir / pydicom.dcmread(dcm).SeriesDescription
            target_folder.mkdir(exist_ok=True)
            dcm.rename(target_folder / dcm.name)

    def create_out_subdirs(self):
        out_subdirs = []
        logging.info("Creating results folders.")
        for subdir in self.in_subdirs:
            to_write = self.out_dir / (subdir.name + "_Results")
            if to_write.exists():
                logging.warning(f"Overwriting results folder: {to_write}")
            else:
                to_write.mkdir()
            out_subdirs.append(to_write)
        return out_subdirs

    def archive_dirs(self):
        archive_dir = self.in_dir / "archived"
        archive_dir.mkdir(exist_ok=True)
        for subdir in [x for x in self.to_archive if x != archive_dir]:
            subdir.rename(archive_dir / subdir.name)

    def enforce_naming_conventions(self):
        for i, subdir in enumerate(self.in_subdirs):
            orientation = substring_matcher(subdir.name, config.EXPECTED_ORIENTATIONS)
            coil = substring_matcher(subdir.name, config.EXPECTED_COILS)
            new_name = subdir.parent / f"{coil}_{orientation}"
            subdir.rename(new_name)
            self.in_subdirs[i] = new_name

    # def restructure_folder_arrangement(self) -> list[Path]:
    #     in_children = []
    #     directories_to_filter = [x for x in self.in_parent.iterdir() if x.is_dir()]
    #     dir_ignored = self.in_parent / "ignored"

    #     nd_filtering_required = any(
    #         "nd" in p.name.lower() for p in directories_to_filter
    #     )
    #     selection_conditions = [
    #         lambda p: len(list(p.iterdir())) == 11,
    #         *([lambda p: "nd" in p.name.lower()] if nd_filtering_required else []),
    #     ]

    #     for p in directories_to_filter:
    #         if all(cond(p) for cond in selection_conditions):
    #             in_children.append(p)
    #         else:
    #             if not dir_ignored.exists():
    #                 dir_ignored.mkdir()
    #             p.rename(dir_ignored / p.name)

    #     return in_children
