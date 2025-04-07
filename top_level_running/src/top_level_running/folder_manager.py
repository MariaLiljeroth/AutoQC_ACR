from pathlib import Path
from tkinter import Tk, filedialog
import pydicom


class FolderManager:

    def run(self):
        self.in_parent = self.get_in_parent()
        self.out_parent = self.get_out_parent()
        self.in_children = self.sort_in_parent()
        self.out_children = self.mk_out_children()

    @staticmethod
    def get_in_parent() -> Path:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        in_parent = Path(
            filedialog.askdirectory(
                parent=root, title="Choose top level input data folder."
            )
        )
        if in_parent.name == "":
            root.destroy()
            raise ValueError("No top level input folder selected.")

        if len(list(in_parent.iterdir())) == 0:
            root.destroy()
            raise ValueError(
                "The selected top level input folder should not be completely empty."
            )

        root.destroy()
        return in_parent

    @staticmethod
    def get_out_parent() -> Path:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        out_parent = Path(
            filedialog.askdirectory(
                parent=root, title="Choose top level output data folder."
            )
        )
        if out_parent.name == "":
            root.destroy()
            raise ValueError("No top level output folder selected.")

        root.destroy()
        return out_parent

    def sort_in_parent(self) -> list[Path]:
        in_children = [p for p in self.in_parent.iterdir() if p.is_dir()]

        if len(in_children) == 0:
            print("Please wait, sorting dicoms...")
            self.sort_dcms_into_dirs(self.in_parent)

        elif all([p.is_dir() for p in in_children]):
            print("Dicoms already sorted.")

        else:
            err = "Toplevel input folder should only contain unsorted dicoms or they should already be sorted."
            raise NotImplementedError(err)

        # Filter out the folders that don't have 11 images inside
        # ND images also filtered out
        in_children = [
            p
            for p in self.in_parent.iterdir()
            if p.is_dir()
            and len(list(p.iterdir())) == 11
            and "nd" not in p.name.lower()
        ]

        return in_children

    @staticmethod
    def sort_dcms_into_dirs(in_parent: Path):
        file_list = [f for f in in_parent.rglob("*") if f.is_file()]

        # Sort each file individually
        for file in file_list:

            # Get SeriesDescription tag
            dcmData = pydicom.dcmread(file)
            sDescrip = dcmData.SeriesDescription

            # Otherwise process the image.
            target_folder = in_parent.joinpath(sDescrip)
            target_folder.mkdir(exist_ok=True)
            file.rename(target_folder.joinpath(file.name))

    def mk_out_children(self):
        out_children = []
        for in_child in self.in_children:
            to_write = self.out_parent / (in_child.name + "_Results")
            if to_write.exists():
                print(f"Overwriting results folder: {to_write}")
            else:
                to_write.mkdir()
                print(f"Generating results folder: {to_write}")
            out_children.append(to_write)
        return out_children

    def get_coil_keys(self):
        coil_keys = []
        in_children_names = [c.name for c in self.in_children]
        for name in in_children_names:
            end_idx = 1
            while (
                sum(name[:end_idx] in other_name for other_name in in_children_names)
                != 1
            ):
                end_idx += 1  # First increment index
            substring = name[: end_idx - 1]  # Then update the substring
            substring = substring.strip("_")
            coil_keys.append(substring)
        return list(set(coil_keys))

    def get_orientation_keys(self):
        orientation_keys = []
        in_children_names = [c.name for c in self.in_children]
        for name in in_children_names:
            end_idx = 1
            while (
                sum(name[-end_idx:] in other_name for other_name in in_children_names)
                != 1
            ):
                end_idx += 1
            substring = name[-end_idx + 1 :]
            substring = substring.strip("_")
            orientation_keys.append(substring)
        return list(set(orientation_keys))
