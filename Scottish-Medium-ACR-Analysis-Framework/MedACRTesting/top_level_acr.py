import sys
sys.path.insert(
    0,
    "C:\\Users\\mliljeroth\\Desktop\\AutoQCtesting\\Scottish-Medium-ACR-Analysis-Framework",
)
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


# import dicom_sorting # Sorting using nathan's script
import os
import shutil

mainpath = askdirectory(title="Choose top level input data folder")
resultspath=askdirectory(title="Choose output data folder")
p=os.listdir(mainpath) # Get string list of all folders
print(p)


# dir = 'path_to_my_folder'
# if os.path.exists(dir):
#     shutil.rmtree(dir)
# os.makedirs(dir)

for u in p:
    print(u)
    ReportDirPath=resultspath+ "/"+ u+"Results"
    if os.path.exists(ReportDirPath):
        print("Results folder exists,overwriting")
        # shutil.rmtree(ReportDirPath)
        # os.makedirs(ReportDirPath)
    else:
        os.makedirs(ReportDirPath)
    #SNR
    acr_snr_task = ACRSNR(input_data=get_dicom_files(mainpath + "/" +u), report_dir=ReportDirPath,report=True,MediumACRPhantom=True)
    snr_dcm = acr_snr_task.ACR_obj.dcms[4] #acr_snr_task.ACR_obj.slice7_dcm 
    snr= acr_snr_task.snr_by_smoothing(snr_dcm)
    print(snr)


    #ST
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