# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 20:31:24 2019

@author: owner
"""

# Convert DICOM to PNG via openCV
import cv2
import os
import pydicom
import glob

home = 'C:\\Users\\owner\\Documents\\PythonScripts\\Kaggle\\Pneumo'
inputdir = 'C:\\Users\\owner\\Documents\\PythonScripts\\Kaggle\\Pneumo\\sample Images'
outdir = 'C:\\Users\\owner\\Documents\\PythonScripts\\Kaggle\\Pneumo\\convertedImages'

test_list = [os.path.basename(x) for x in glob.glob(inputdir + './*.dcm')]

os.chdir(outdir)

for f in test_list:  
    print(f)
    ds = pydicom.read_file(inputdir +'\\' + f) # read dicom image
    img = ds.pixel_array # get image array
    cv2.imwrite(f.replace('.dcm','.png'),img) # write png image

os.chdir(home)