import torch
import numpy as np
from scipy.ndimage import label as label
import torchvision.transforms as transforms


def Bbox_to_arr(bbox):
    
    arr = np.zeros([270, 480, 1])
    
    topleft_y =  bbox[1]
    topleft_x = bbox[0]
    bottright_y = bbox[3]
    bottright_x = bbox[2]

    min_x = np.amin([topleft_x,bottright_x])
    max_x = np.amax([topleft_x,bottright_x])
    min_y = np.amin([topleft_y,bottright_y])
    max_y = np.amax([topleft_y,bottright_y])
        
    arr[min_y:max_y,min_x:max_x,:] = 1
    arr = transforms.functional.to_tensor(np.array(arr))
    return arr

#https://www.geeksforgeeks.org/find-two-rectangles-overlap/
def intersection_of_bboxes(bbox_1,bbox_2):
    
    if (bbox_1[0] == bbox_1[2] or bbox_2[0]== bbox_2[2] or bbox_1[1]==bbox_1[3] or  bbox_2[1]==bbox_2[3]):
        # bounding boxok vonalak
        return False
    
    if (bbox_1[1]>=bbox_2[3] or bbox_2[1] >= bbox_1[3]):
        # egymás felett
        return False
    
    if (bbox_1[0]>bbox_2[2] or bbox_2[0]>bbox_2[2]):
        # egymás mellett
        return False
    
    return True

def intersection_of_arrays(target,pred):
    IS = target*pred
    nnz = torch.count_nonzero(IS)
    if nnz>0:
        return True
    else:
        return False
    
def N_false_pos(target,pred):
    diff = pred-target
    _,N = label(diff.detach().numpy())
    return N

def fp_pixels(target,pred):
    diff = pred - target
    nnz = torch.count_nonzero((diff>0))
    return nnz

def tp_pixels(target,pred):
    IS = target*pred
    nnz = torch.count_nonzero(IS)
    return nnz

def fn_pixels(target,pred):
    diff = target - pred
    nnz = torch.count_nonzero(diff<0)
    return nnz

def tn_pixels(target,pred):
    IS = target*pred
    diff = target - pred
    indic = diff-IS
    nnz = torch.count_nonzero((indic==0))
    return nnz

def tpr(target,pred):
    fn = fn_pixels(target,pred)
    tp = tp_pixels(target,pred)
    tp_rate = tp/(tp+fn)
    return tp_rate.item()

def fpr(target,pred):
    fp = fp_pixels(target,pred)
    tn = tn_pixels(target,pred)
    fp_rate = fp/(tn+fp)
    return fp_rate.item()


    
