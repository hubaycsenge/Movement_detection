import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data
from functools import partial
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from custom_losses import IoULoss
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
from scipy.ndimage import label
from scipy.ndimage.measurements import center_of_mass

def midpoint(mask):

	lab_mask,feat_num = label(mask)
	max_nnz = 0
	max_nnz_num = 0
	for i in range(1,feat_num+1):
		nnz = np.count_nonzero(lab_mask == i)
		if nnz > max_nnz_num:
			max_nnz = i
			max_nnz_num = nnz
	mass = (lab_mask == max_nnz)*1
	print(mass.shape)
	center_coords = center_of_mass(mass)
	return center_coords

size = (1080,1920)
cap = cv2.VideoCapture('inputs/track_validation_vid.mp4')
capwrite = cv2.VideoCapture('outputs/track_validation_vid.mp4')
out = cv2.VideoWriter('outputs/yolov5tracking_uballtracking_done.mp4',cv2.VideoWriter_fourcc(*'XVID'), 30, (size[1],size[0]))

def superimpose_mask(image_array, mask_array, opacity=1, color_index=0, grayscale=True):
	'''
	superimpose the nodule mask on the CT segment with adjustable opacity level
	color index = 0, 1 and 2 indicates color red, green and blue in order
	'''
	print(image_array.shape,mask_array.shape)
	if grayscale:
		superimposed = gray_to_colored(image_array)
	else:
		superimposed = image_array.copy()
		
	colored_mask = np.zeros(image_array.shape)
	colored_mask[:, :,color_index] = mask_array == 1
	colored_mask = colored_mask.astype(np.bool)
	superimposed[colored_mask] = opacity * 1 + (1 - opacity) * superimposed[colored_mask]
	return superimposed

model = UNet(in_channels=3,
             out_channels=1,
             n_blocks=2,
             start_filters=32,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=2)
criterion = IoULoss()


model.load_state_dict(torch.load("otf_model_net_epoch41_loss=0.4606550086411276.pt",map_location = torch.device('cpu')))
center_coords = []
while(cap.isOpened() or capwrite.isOpened()):
	ret, frame = cap.read()
	ret,writeframe = capwrite.read()
	frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
	inputs = transforms.functional.to_tensor(np.array(frame))
	inputs = torch.stack((inputs, inputs, inputs, inputs, inputs, inputs, inputs, inputs),dim = 0)
	outputs = model(inputs)
	pred = outputs[7].cpu().detach().numpy()
	pred[pred<0.15] = 0
	pred[pred>0.15] = 1
	pred = np.moveaxis(pred,0,-1)
	frame = cv2.resize(frame, (0,0), fx=4, fy=4)
	pred =  cv2.resize(pred, (0,0), fx=4, fy=4)
	SI = superimpose_mask(writeframe,pred,color_index = 0,grayscale=False) 
	y,x = midpoint(pred)
	print(x,y)
	for dot in center_coords:
		SI = cv2.circle(SI, (round(dot[0]),round(dot[1])), radius=2, color=(0, 0, 255), thickness=-1)
	print('ASD')	
	SI = cv2.circle(SI, (round(x),round(y)), radius=5, color=(0, 0, 255), thickness=-1)
	out.write(SI)
	center_coords.append((x,y))
	
	cv2.imshow('prediction',SI)
	print('ASD')	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
capwrite.release()
out.release()
cv2.destroyAllWindows()
