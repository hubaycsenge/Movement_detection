import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.utils.data import Dataset,DataLoader

from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

import copy
from collections import namedtuple
import os
import random
import shutil
import time

import cv2
import zipfile
import json
import gc

from unet import UNet


class DatasetWithCorners(Dataset):

    def __init__(self,folder, json, transforms=None):

        self.whole_dict = json
        self.folder = folder
        self.transforms = transforms
        self.png_paths = json
        self.fnames = []
        for key in list(json.keys()):
            self.fnames.append(key)
            
        self.fnames.sort()


    def __len__(self):
        return len(self.png_paths)


    def __getitem__(self,idx):
        
        fname = self.fnames[idx]
        png_path = os.path.join(self.folder,fname)

        png = cv2.imread(png_path)
        png = cv2.resize(png, (0,0), fx=0.25, fy=0.25) 
        image = transforms.functional.to_tensor(np.array(png))

        if self.transforms:
              image = self.transforms(image)
        #print('shape',image.shape)
        label = [self.png_paths[fname]['TOPLEFT_X']/(image.shape[2]*4),self.png_paths[fname]['TOPLEFT_Y']/(image.shape[1]*4),self.png_paths[fname]['BOTTOMRIGHT_X']/(image.shape[2]*4),self.png_paths[fname]['BOTTOMRIGHT_Y']/(image.shape[1]*4)]
        #print(fname, label)
        return image, torch.FloatTensor(label),fname
    
    
class DatasetWithMask(Dataset):

    def __init__(self,folder, json, transforms=None):

        self.whole_dict = json
        self.folder = folder
        self.transforms = transforms
        self.png_paths = json
        self.fnames = []
        for key in list(json.keys()):
            self.fnames.append(key)
            
        self.fnames.sort()


    def __len__(self):
        return len(self.png_paths)


    def __getitem__(self,idx):
        
        fname = self.fnames[idx]
        png_path = os.path.join(self.folder,fname)

        png = cv2.imread(png_path)
        png = cv2.resize(png, (0,0), fx=0.25, fy=0.25) 
        image = transforms.functional.to_tensor(np.array(png))

        if self.transforms:
              image = self.transforms(image)

        topleft_y = round(self.png_paths[fname]['TOPLEFT_Y']/4)
        topleft_x = round(self.png_paths[fname]['TOPLEFT_X']/4)
        bottright_y = round(self.png_paths[fname]['BOTTOMRIGHT_Y']/4)
        bottright_x = round(self.png_paths[fname]['BOTTOMRIGHT_X']/4)
        
        min_x = np.amin([topleft_x,bottright_x])
        max_x = np.amax([topleft_x,bottright_x])
        min_y = np.amin([topleft_y,bottright_y])
        max_y = np.amax([topleft_y,bottright_y])
        
        label = np.zeros([270, 480, 1])
        label[min_y:max_y,min_x:max_x,:] = 1
        label = transforms.functional.to_tensor(np.array(label))
        
        #print(fname, label)
        return image, label,fname
    
def superimpose_mask(image_array, mask_array, opacity=1, color_index=0, grayscale=True):
    '''
    superimpose the nodule mask on the CT segment with adjustable opacity level
    color index = 0, 1 and 2 indicates color red, green and blue in order
    '''
    if grayscale:
        superimposed = gray_to_colored(image_array)
    else:
        superimposed = image_array.copy()
        
    colored_mask = np.zeros(image_array.shape)
    colored_mask[:, :,color_index] = mask_array[:,:,0] == 1
    colored_mask = colored_mask.astype(np.bool)
    superimposed[colored_mask] = opacity * 1 + (1 - opacity) * superimposed[colored_mask]
    return superimposed
