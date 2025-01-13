import os
import sys

sys.path.insert(
    0,
    "C:\\Users\\mliljeroth\\Desktop\\AutoQCtesting\\Scottish-Medium-ACR-Analysis-Framework",
)
sys.path.append(os.path.dirname(os.getcwd()))

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")

from tkinter.filedialog import askdirectory

from hazenlib.utils import get_dicom_files, sortDICOMs
from hazenlib.tasks.acr_slice_thickness import ACRSliceThickness
from hazenlib.tasks.acr_snr import ACRSNR


class TopLevel:
    def run(self):
        (
            self.mainPath,
            self.resultsPath,
            self.foldersInMainPath,
            self.reportDirPath,
        ) = self.create_IO()
        for folder in self.foldersInMainPath:
            self.run_tasks_on_folder(folder)

    def create_IO(self) -> tuple[str, str, list[str], str]:
        # Select I/O folders
        mainPath = askdirectory(title="Choose top level input data folder")
        resultsPath = askdirectory(title="Choose output data folder")

        # Find folders in mainPath and sort Input folder using DICOM sorter.
        foldersInMainPath = [
            item for item in os.listdir(mainPath) if os.path.isdir(os.path.join(mainPath, item))
        ]
        if len(foldersInMainPath) == 0:
            sortDICOMs(mainPath)
            foldersInMainPath = [
                item for item in os.listdir(mainPath) if os.path.isdir(os.path.join(mainPath, item))
            ]

        # Generate results folder inside of Ouput folder.
        for folder in foldersInMainPath:
            reportDirPath = resultsPath + "/" + folder + "Results"
            if os.path.exists(reportDirPath):
                print("Results folder exists, overwriting")
            else:
                os.makedirs(reportDirPath)

        return mainPath, resultsPath, foldersInMainPath, reportDirPath

    def run_tasks_on_folder(self, folder: str):
        acr_snr_task = ACRSNR(
            input_data=get_dicom_files(self.mainPath + "/" + folder),
            report_dir=self.reportDirPath,
            report=True,
            MediumACRPhantom=True,
        )
        acr_snr_task.run()

        # snr_dcm = acr_snr_task.ACR_obj.dcms[
        #    4
        # ]  # acr_snr_task.ACR_obj.slice7_dcm
        # snr = acr_snr_task.snr_by_smoothing(snr_dcm)
        # print(snr)


toplevel = TopLevel()
toplevel.run()


# dir = 'path_to_my_folder'
# if os.path.exists(dir):
#     shutil.rmtree(dir)
# os.makedirs(dir)

# SNR


# ST
# acr_slice_thickness_task = ACRSliceThickness(
# input_data= get_dicom_files(mainpath + "/" +u), report_dir=ReportDirPath, report=True, MediumACRPhantom=True
# )
# dcm = acr_slice_thickness_task.ACR_obj.dcms[0] # not currently true for sag slices where slice order is reversed so slice thickness would be dcms[end]
# SliceThick = acr_slice_thickness_task.get_slice_thickness(dcm)
# print(SliceThick)


# tasks= ["ST", "SNR", "UNI", "GT", "RES"]

# acr_slice_thickness_task = ACRSliceThickness(
#     input_data=files, report_dir=ReportDirPath, report=True, MediumACRPhantom=True
# )
# dcm = acr_slice_thickness_task.ACR_obj.dcms[0]
# SliceThick = acr_slice_thickness_task.get_slice_thickness(dcm)
# print(SliceThick)
