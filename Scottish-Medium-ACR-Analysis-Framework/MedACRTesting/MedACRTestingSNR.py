# import sys
# # sys.path.insert(0,"C:\\Users\\Johnt\\Documents\\GitHub\\Hazen-ScottishACR-Fork")
# # sys.path.insert(0,"D:\\Hazen-ScottishACR-Fork")
# import pydicom
# from hazenlib.utils import get_dicom_files
# from hazenlib.tasks.acr_snr import ACRSNR
# from hazenlib.tasks.acr_uniformity import ACRUniformity
# from hazenlib.ACRObject import ACRObject
# import pathlib
# from tests import TEST_DATA_DIR, TEST_REPORT_DIR

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
from hazenlib.tasks.acr_snr import ACRSNR
from hazenlib.ACRObject import ACRObject
import pathlib
# from tests import TEST_DATA_DIR, TEST_REPORT_DIR
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory




ReportDirPath = "MedACRTests"
'''
#Make sure it works with the test data
files = get_dicom_files("tests\\data\\acr\\GE")
acr_snr_task = ACRSNR(input_data=files, report_dir=ReportDirPath,report=True)
snr_dcm = acr_snr_task.ACR_obj.dcms[6]
snr, _ = acr_snr_task.snr_by_smoothing(snr_dcm)
print(snr)
'''

files = get_dicom_files(askdirectory())
# ACRDICOMSFiles = {}
# for file in files:
#     data = pydicom.dcmread(file)
#     if (data.SeriesDescription not in ACRDICOMSFiles.keys()):
#         ACRDICOMSFiles[data.SeriesDescription]=[]
#     ACRDICOMSFiles[data.SeriesDescription].append(file)
# ChosenData = ACRDICOMSFiles["ACR AxT1"]



#Test SNR
#Only change neede was the paramaters of the hough circles
acr_snr_task = ACRSNR(input_data=files, report_dir=ReportDirPath,report=True,MediumACRPhantom=True)
snr_dcm = acr_snr_task.ACR_obj.slice7_dcm #acr_snr_task.ACR_obj.dcms[6]
snr, _ = acr_snr_task.snr_by_smoothing(snr_dcm,5)
print(snr)

