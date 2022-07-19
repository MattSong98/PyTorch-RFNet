import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

from rfnet.utils.transforms import *


def get_mask_combinations():

    masks = [[False, False, False, True], 
            [False, True, False, False], 
            [False, False, True, False], 
            [True, False, False, False],
            [False, True, False, True], 
            [False, True, True, False], 
            [True, False, True, False], 
            [False, False, True, True], 
            [True, False, False, True], 
            [True, True, False, False],
            [True, True, True, False], 
            [True, False, True, True], 
            [True, True, False, True], 
            [False, True, True, True],
            [True, True, True, True]]

    masks_torch = torch.from_numpy(np.array(masks))

    mask_name = ['t2', 't1c', 't1', 'flair', 
                 't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
                 'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
                 'flairt1cet1t2']

    return masks, masks_torch, mask_name


def get_mask_combinations_exp1():

    masks = [[False, False, False, True], 
            [False, True, False, False], 
            [False, False, True, False], 
            [False, True, False, True], 
            [False, True, True, False], 
            [False, False, True, True], 
            [False, True, True, True]]

    masks_torch = torch.from_numpy(np.array(masks))

    mask_name = ['t2', 't1c', 't1', 
                 't1cet2', 't1cet1', 't1t2',
                 't1cet1t2']

    return masks, masks_torch, mask_name


class Brats_loadall(Dataset):

    def __init__(self, transforms='', root=None, num_cls=4, train_file='train.txt', mask_generator=get_mask_combinations):

        data_file_path = os.path.join(root, train_file) 
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()]
        datalist.sort()

        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))

        masks, _, _ = mask_generator()
        
        self.names = datalist      
        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.mask_array = np.array(masks)
        self.num_cls = num_cls

    def __getitem__(self, index):

        name = self.names[index]
        volpath = self.volpaths[index]
        segpath = volpath.replace('vol', 'seg')
        
        x = np.load(volpath) 
        y = np.load(segpath).astype(np.uint8) 
        x, y = x[None, ...], y[None, ...] 

        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))

        _, H, W, Z = np.shape(y)
        y = np.reshape(y, (-1))
        one_hot_targets = np.eye(self.num_cls)[y]
        yo = np.reshape(one_hot_targets, (1, H, W, Z, -1))
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))

        x = torch.squeeze(torch.from_numpy(x), dim=0)
        yo = torch.squeeze(torch.from_numpy(yo), dim=0)

        num_masks = len(mask_array)
        mask_idx = int(np.random.choice(num_masks, 1)) 
        mask = torch.from_numpy(self.mask_array[mask_idx])
        return x, yo, mask, name

    def __len__(self):
        return len(self.volpaths)


class Brats_loadall_eval(Dataset):

    def __init__(self, transforms='', root=None, test_file='test.txt'):

        data_file_path = os.path.join(root, test_file)
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()]
        datalist.sort()

        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname + '_vol.npy'))
        
        self.names = datalist
        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')

    def __getitem__(self, index):

        name = self.names[index]
        volpath = self.volpaths[index]
        segpath = volpath.replace('vol', 'seg')

        x = np.load(volpath)
        y = np.load(segpath).astype(np.uint8)
        x, y = x[None, ...], y[None, ...]

        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)

        x = torch.squeeze(torch.from_numpy(x), dim=0)
        y = torch.squeeze(torch.from_numpy(y), dim=0)

        return x, y, name

    def __len__(self):
        return len(self.volpaths)


# worker_init_fn (callable, optional) – If not None, 
# this will be called on each worker subprocess with the worker id (an int in [0, num_workers - 1]) as input, 
# after seeding and before data loading. (default: None)

def init_fn(worker):
    M = 2**32 - 1
    seed = torch.LongTensor(1).random_().item()
    seed = (seed + worker) % M
    np.random.seed(seed)
    random.seed(seed)

