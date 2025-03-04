import os
import sys

sys.path.insert(
    0,
    "C:\\Users\\mliljeroth\\Desktop\\AutoQCtesting\\Scottish-Medium-ACR-Analysis-Framework",
)
sys.path.append(os.path.dirname(os.getcwd()))

from tkinter import Tk, filedialog
from typing import Union

from hazenlib.utils import get_dicom_files, sortDICOMs
from hazenlib.tasks.acr_slice_thickness import ACRSliceThickness
from hazenlib.tasks.acr_uniformity import ACRUniformity
from hazenlib.tasks.acr_snr import ACRSNR
from hazenlib.tasks.acr_geometric_accuracy import ACRGeometricAccuracy
from hazenlib.tasks.acr_spatial_resolution_rsch_test import ACRSpatialResolution

class TopLevel:
    folders_to_exclude = ["_Excluded Images", "_ND Images"]
    task_to_class_mapping = {
        "slice_thickness": ACRSliceThickness,
        "snr": ACRSNR,
        "geometric_distortion": ACRGeometricAccuracy,
        "uniformity": ACRUniformity,
        "spatial_resolution": ACRSpatialResolution
        }
    class_str_to_task_mapping = {val.__name__: key for key, val in task_to_class_mapping.items()}

    def run(self, tasks: list):
        self.tasks = tasks
        self.create_IO()
        self.results = self.get_results()
        pass




    def create_IO(self) -> tuple[str, str, list[str], list[str]]:
        self.in_top = self._get_in_top_dir()
        self.out_top = self._get_out_top_dir()
        self.in_dirs = self._sort_dicoms()
        self.out_dirs = self._gen_out_dirs()

    @staticmethod
    def _get_in_top_dir():
        root = Tk()
        root.withdraw()
        in_top = filedialog.askdirectory(parent=root, title="Choose top level input data folder.")
        if in_top == "":
            root.destroy()
            raise ValueError("No top level input folder selected.")
        if len(os.listdir(in_top)) == 0:
            root.destroy()
            raise ValueError("The selected top level input folder should not be completely empty.")
        return in_top

    @staticmethod
    def _get_out_top_dir():
        root = Tk()
        root.withdraw()
        out_top = filedialog.askdirectory(
            parent=root, title="Choose top level output data folder."
        )
        if out_top == "":
            root.destroy()
            raise ValueError("No top level output folder selected.")
        root.destroy()
        return out_top

    def _sort_dicoms(self):
        in_dirs = [
            item for item in os.listdir(self.in_top) if os.path.isdir(os.path.join(self.in_top, item))
        ]
        if len(in_dirs) == 0:
            sortDICOMs(self.in_top)
            in_dirs = [
                item for item in os.listdir(self.in_top) if os.path.isdir(os.path.join(self.in_top, item))
            ]
        elif len(in_dirs) == 9 + len(self.folders_to_exclude):
            pass
        else:
            raise ValueError(
                "There should only be DICOM images in the top level input folder when running this programme, \n or the images should already be sorted."
            )
        in_dirs = [os.path.join(self.in_top, x) for x in in_dirs if not any([x in folder for folder in self.folders_to_exclude])]
        return in_dirs

    def _gen_out_dirs(self):
        out_dirs = []
        for in_dir in self.in_dirs:
            if not any(x in in_dir for x in self.folders_to_exclude):
                out_dir = f"{self.out_top}/{os.path.basename(in_dir)}Results"
                if os.path.exists(out_dir):
                    print("Results folder exists, overwriting results.")
                else:
                    os.makedirs(out_dir)
                out_dirs.append(out_dir)
        return out_dirs

    def get_results(self):
        def _extract_coil(file: Union[str, list]):
            if isinstance(file, list):
                file = file[0]
            file = os.path.basename(file)
            return file[:file.find("_")]

        # init results dict
        tasks_keys = [f"{task}" for task in self.task_to_class_mapping]
        coils_keys = set([_extract_coil(d) for d in self.out_dirs])
        results_dict = {tk: {ck: [] for ck in coils_keys} for tk in tasks_keys}

        # get results and assign to results dict so sorted by task and then coil
        results = [self._run_tasks_on_folder(in_dir, out_dir, tasks = self.tasks) for (in_dir, out_dir) in zip(self.in_dirs, self.out_dirs)]
        for outer_list in results:
            for inner_dict in outer_list:
                coil_key = _extract_coil(inner_dict["file"])
                task_key = self.class_str_to_task_mapping[inner_dict["task"]]
                results_dict[task_key][coil_key] = inner_dict
        return results_dict

    def _run_tasks_on_folder(self, in_dir: str, out_dir: str, tasks: list) -> list[dict]:
        input_data = get_dicom_files(os.path.join(in_dir))
        task_objs = [
            self.task_to_class_mapping[task](input_data=input_data, report_dir=out_dir, report=True, MediumACRPhantom=True)
            for task in tasks
            ]
        results = [taskobj.run() for taskobj in task_objs]
        return results


toplevel = TopLevel()
toplevel.run(tasks = [
        "slice_thickness",
        "snr",
        "geometric_distortion",
        "uniformity"
    ])
