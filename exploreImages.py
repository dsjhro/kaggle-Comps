# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:05:01 2019

@author: owner
"""

import pandas as pd
import numpy as np
from mask_functions import rle2mask
import os
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image
from fastai import vision as vs
import cv2


path = os.getcwd()
#data = path  + '\\'  + "data" + '\\' + "train-rle.csv"
data = path + '\\' + "sample Images" + "\\" + "train-rle-sample.csv"

annoteDF = pd.read_csv(data, delimiter = ",")
id0 = annoteDF['id'][5]
rle0 = annoteDF['rle'][5]

#id0 = annoteDF['ImageId'][2]
#rle0 = annoteDF['EncodedPixels'][2]


imgpath = path + "\\" + "sample Images" + "\\" + "sampleConvertedImages" + "\\" + id0 +".png"

img = Image.open(imgpath)
w,h = img.size

# mask with fastai
'''
    1) grab csv rle label 
    2) label = rle_encode(csv rle label)
    3) mask = open_mask_rle(label, shape=(1024, 1024)).resize((1,1024,1024)))


'''

test = rle2mask(rle0, w,h)
label = vs.rle_encode(test)
mask = vs.open_mask_rle(label, shape=(1024, 1024)).resize((1,1024,1024))

w = 1024
h = 1024

def getLabelFunc(x):
    
    if(len(annoteDF['rle'][x])<3):
        #return(vs.open_mask_rle('', shape=(1024, 1024)).resize((1,1024,1024)))
        return(np.zeros([1024,1024]))
    else:
        temp = rle2mask(annoteDF['rle'][x], 1024,1024)
        return(temp)
        
        #label = vs.rle_encode(temp)
        #mask = vs.open_mask_rle(label, shape=(1024, 1024)).resize((1,1024,1024))
        #return(mask)

for entry in range(0,len(annoteDF['rle'])):
    temp = getLabelFunc(entry)
    fileName = annoteDF['id'][entry] + '.png'
    cv2.imwrite(fileName, temp)

'''
Image.fromarray(test)

classes = ["Negative", "postive"]

plt.imshow(test, interpolation='nearest')
plt.imshow(img, interpolation='nearest', alpha = 0.5)
plt.show()


image.imsave('name.png', test)
mask = open_mask('name.png')
'''