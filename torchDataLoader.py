# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
import numpy as np
from PIL import Image
import os
import pandas as pd
from mask_functions import rle2mask
import torchvision

class lungData(Dataset):
    
    def __init__(self, root, size = (224,224), transform = None):
        self.files = glob.glob(root+'\\*.png')
        self.labdf = pd.read_csv(glob.glob(root+'\\*.csv')[0])
        self.size = size
        self.transform = transform
        
    def __len__(self):
        return len(self.files)
    
    def getLabelFunc(self, x):
        
         if(len(x)<3):
             return(np.zeros(self.size))
         else:
             temp = rle2mask(x, self.size[0],self.size[1])
             return(temp)
    
    def __getitem__(self,idx):
        img = np.asarray(Image.open(self.files[idx]).resize(self.size))
        rle = self.labdf['rle'][idx]
        labels = self.getLabelFunc(rle)
        ids = self.labdf['id'][idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img,labels,ids
    
data_folder = os.getcwd() + "\\Sample Images\\sampleConvertedImages"
lungset = lungData(data_folder)
data = DataLoader(lungset,batch_size=2, shuffle = True)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2


'''
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms, datasets
data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])


def getLabelFunc(x):
    
    if(len(annoteDF['rle'][x])<3):
        #return(vs.open_mask_rle('', shape=(1024, 1024)).resize((1,1024,1024)))
        return(np.zeros([1024,1024]))
    else:
        temp = rle2mask(annoteDF['rle'][x], 1024,1024)
        return(temp)
        
        '''