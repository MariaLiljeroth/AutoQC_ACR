import pydicom as pyd
import tkinter as tk
from tkinter.filedialog import askdirectory
import pathlib
import os


class Main:
    def __init__(self):
        self.dirPath = self.selectDirectory()
        self.sortDICOMS()

    def selectDirectory(self):
        """
        Description
        ------
        Asks the user to select a directory of unsorted DICOM images

        Returns
        ------
        dirPath: pathlib.Path
            A Windows path to the user selected directory.
        """

        root = tk.Tk()
        root.withdraw()
        root.call("wm", "attributes", ".", "-topmost", True)
        windowTitle = "Select a Folder for your Data Analysis"
        dirPath = askdirectory(title=windowTitle, parent=root)
        dirPath = pathlib.Path(dirPath)
        root.destroy()

        return dirPath

    def sortDICOMS(self):
        """
        Description
        ------
        Groups the DICOM files within the selected directory into folders - one series per folder
        """

        fileList = [item for item in self.dirPath.rglob("*") if item.is_file()]

        for file in fileList:
            fileName = os.path.split(file)[1]

            dcmData = pyd.dcmread(file)
            sDescrip = dcmData.SeriesDescription

            folderPath = str(self.dirPath) + "/" + sDescrip
            if not os.path.exists(folderPath):
                os.makedirs(folderPath)

            os.rename(file, folderPath + "/" + fileName)


main = Main()
