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
from hazenlib.tasks.acr_geometric_accuracy import ACRGeometricAccuracy
from hazenlib.tasks.acr_spatial_resolution import ACRSpatialResolution


class TopLevel:
    def run(self):
        self.foldersToExclude = ["_Excluded Images", "_ND Images"]
        (
            self.mainPath,
            self.resultsPath,
            self.foldersInMainPath,
            self.outputFolders,
        ) = self.create_IO()

        tests = ["sth", "snr", "uni", "geo", "res"]
        keys = [f"{test}Grouping" for test in tests]
        groups = {key: [] for key in keys}
                  
        for folder, outputFolder in zip(self.foldersInMainPath, self.outputFolders):
            results = self.run_tasks_on_folder(folder, outputFolder)
            for i, result in enumerate(results):
                groups[keys[i]].append(result)
        
        pass
                
    def create_IO(self) -> tuple[str, str, list[str], list[str]]:
        # Select I/O folders
        root = Tk()
        root.withdraw()

        mainPath = filedialog.askdirectory(parent=root, title="Choose top level input data folder.")
        if mainPath == "":
            raise ValueError("No top level input folder selected.")
        if len(os.listdir(mainPath)) == 0:
            root.destroy()
            raise ValueError("The selected top level input folder should not be completely empty.")

        resultsPath = filedialog.askdirectory(
            parent=root, title="Choose top level output data folder."
        )
        if resultsPath == "":
            raise ValueError("No top level output folder selected.")

        # Find folders in mainPath and sort Input folder using DICOM sorter.
        foldersInMainPath = [
            item for item in os.listdir(mainPath) if os.path.isdir(os.path.join(mainPath, item))
        ]

        # If completely unsorted then sort.
        if len(foldersInMainPath) == 0:
            sortDICOMs(mainPath)
            foldersInMainPath = [
                item for item in os.listdir(mainPath) if os.path.isdir(os.path.join(mainPath, item))
            ]
        elif len(foldersInMainPath) == 9 + len(self.foldersToExclude):
            pass
        else:
            raise ValueError(
                "There should only be DICOM images in the top level input folder when running this programme, \n or the images should already be sorted."
            )

        # Generate results folders inside of toplevel output folder and record paths.
        outputFolders = []
        for folder in foldersInMainPath:
            if True not in [x in folder for x in self.foldersToExclude]:
                reportDirPath = resultsPath + "/" + folder + "Results"
                if os.path.exists(reportDirPath):
                    print("Results folder exists, overwriting results.")
                else:
                    os.makedirs(reportDirPath)
                outputFolders.append(reportDirPath)

        return mainPath, resultsPath, foldersInMainPath, outputFolders

    def run_tasks_on_folder(self, folder: str, outputFolder: str) -> list[dict]:
        if True not in [x in folder for x in self.foldersToExclude]:
            input_data = get_dicom_files(os.path.join(self.mainPath, folder))


            stTask = ACRSliceThickness(
                input_data=input_data, report_dir=outputFolder, report=True, MediumACRPhantom=True
            )

            snrTask = ACRSNR(
                input_data=input_data, report_dir=outputFolder, report=True, MediumACRPhantom=True
            )
            
            geomTask = ACRGeometricAccuracy(
                input_data=input_data, report_dir=outputFolder, report=True, MediumACRPhantom=True
            )
            
            unifTask = ACRUniformity(
                input_data=input_data, report_dir=outputFolder, report=True, MediumACRPhantom=True
            )
            
            """
            resTask = ACRSpatialResolution(
                input_data=input_data, report_dir=outputFolder, report=True, MediumACRPhantom=True
            )
            """


            stResults = stTask.run()
            snrResults = snrTask.run()
            geomResults = geomTask.run()
            unifResults = unifTask.run()
            #resResults = resTask.run()
            
            #stResults = None
            #snrResults = None
            #geomResults = None
            #unifResults = None
            resResults = None

            return stResults, snrResults, unifResults, geomResults, resResults

        else:
            return ['Excluded' for x in range(5)]


toplevel = TopLevel()
toplevel.run()
