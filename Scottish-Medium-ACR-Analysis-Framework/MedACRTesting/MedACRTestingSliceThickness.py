import sys

sys.path.insert(
    0,
    "C:\\Users\\mliljeroth\\Desktop\\AutoQCtesting\\Scottish-Medium-ACR-Analysis-Framework",
)
# sys.path.insert(0,"D:\\Hazen-ScottishACR-Fork")
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pydicom
from hazenlib.utils import get_dicom_files
from hazenlib.tasks.acr_slice_thickness import ACRSliceThickness
from hazenlib.ACRObject import ACRObject
import pathlib
# from tests import TEST_DATA_DIR, TEST_REPORT_DIR
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory

ReportDirPath = "MedACRTests"
"""
#Make sure it works with the test data
files = get_dicom_files("tests\\data\\acr\\GE")
acr_slice_thickness_task = ACRSliceThickness(input_data=files,report_dir=ReportDirPath,report=True)
dcm = acr_slice_thickness_task.ACR_obj.dcms[0]
SliceThick = acr_slice_thickness_task.get_slice_thickness(dcm)
print(SliceThick)
"""

# files = get_dicom_files("ACR_MRI1_20240116_104902641")
files = get_dicom_files(askdirectory())
ACRDICOMSFiles = {}
for file in files:
    data = pydicom.dcmread(file)
    if data.SeriesDescription not in ACRDICOMSFiles.keys():
        ACRDICOMSFiles[data.SeriesDescription] = []
    ACRDICOMSFiles[data.SeriesDescription].append(file)
    print(ACRDICOMSFiles)
# ChosenData = ACRDICOMSFiles["ACR AxT1"]
ChosenData = ACRDICOMSFiles["FlexSp45_Ax"]



acr_slice_thickness_task = ACRSliceThickness(
    input_data=ChosenData, report_dir=ReportDirPath, report=True, MediumACRPhantom=True
)
dcm = acr_slice_thickness_task.ACR_obj.dcms[0]
SliceThick = acr_slice_thickness_task.get_slice_thickness(dcm)
print(SliceThick)
