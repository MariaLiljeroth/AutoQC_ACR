import os
import sys

sys.path.insert(
    0,
    "C:\\Users\\mliljeroth\\Desktop\\AutoQCtesting\\Scottish-Medium-ACR-Analysis-Framework",
)
sys.path.append(os.path.dirname(os.getcwd()))

from tkinter import Tk, filedialog

from hazenlib.utils import get_dicom_files, sortDICOMs
from hazenlib.tasks.acr_slice_thickness_rsch import ACRSliceThickness
from hazenlib.tasks.acr_uniformity import ACRUniformity
from hazenlib.tasks.acr_snr import ACRSNR


class TopLevel:
    def run(self):
        (
            self.mainPath,
            self.resultsPath,
            self.foldersInMainPath,
            self.outputFolders,
        ) = self.create_IO()
        
        for folder, outputFolder in zip(self.foldersInMainPath, self.outputFolders):
            self.run_tasks_on_folder(folder, outputFolder)

    def create_IO(self) -> tuple[str, str, list[str], list[str]]:
        # Select I/O folders
        root = Tk()
        mainPath = filedialog.askdirectory(parent=root, title="Choose top level input data folder")
        resultsPath = filedialog.askdirectory(parent=root, title="Choose output data folder")
        root.destroy()

        # Find folders in mainPath and sort Input folder using DICOM sorter.
        foldersInMainPath = [
            item for item in os.listdir(mainPath) if os.path.isdir(os.path.join(mainPath, item))
        ]
        if len(foldersInMainPath) == 0:
            sortDICOMs(mainPath)
            foldersInMainPath = [
                item for item in os.listdir(mainPath) if os.path.isdir(os.path.join(mainPath, item))
            ]

        # Generate results folders inside of toplevel output folder and record paths.
        outputFolders = []
        for folder in foldersInMainPath:
            reportDirPath = resultsPath + "/" + folder + "Results"
            if os.path.exists(reportDirPath):
                print("Results folder exists, overwriting results.")
            else:
                os.makedirs(reportDirPath)
            outputFolders.append(reportDirPath)

        return mainPath, resultsPath, foldersInMainPath, outputFolders

    def run_tasks_on_folder(self, folder: str, outputFolder: str):
        input_data = get_dicom_files(os.path.join(self.mainPath, folder))
        
        stTask = ACRSliceThickness(
            input_data=input_data,
            report_dir=outputFolder,
            report=True,
            MediumACRPhantom=True
        )
        
        snrTask = ACRSNR(
            input_data=input_data,
            report_dir=outputFolder,
            report=True,
            MediumACRPhantom=True)
        
        uniTask = ACRUniformity(
            input_data=input_data,
            report_dir=outputFolder,
            report=True,
            MediumACRPhantom=True)
        
        stResults = stTask.run()
        snrResults = snrTask.run()
        uniResults = uniTask.run()

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
