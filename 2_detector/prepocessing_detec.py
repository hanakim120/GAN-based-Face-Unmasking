import os
import torch
import torch.nn as nn
import torch.utils.data as data
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image

class FacemaskSegDataset(data.Dataset):
    
    def __init__(self, cfg, train=True):
        
        
    def load_images(self):
        self.fns = []
        for idx, rows in self.df.iterrows():
            _ , img_name, mask_name = rows
            img_path = os.path.join(self.root_dir, img_name)
            mask_path = os.path.join(self.root_dir, mask_name)
            img_path = img_path.replace('\\','/')
            mask_path = mask_path.replace('\\','/')
            if os.path.isfile(mask_path): 
                self.fns.append([img_path, mask_path])

        

    
    def collate_fn(self, batch):
        imgs = torch.stack([i[0] for i in batch])
        masks = torch.stack([i[1] for i in batch])
        return {
            'imgs': imgs,
            'masks': masks
        }
    
    def __len__(self):
        return len(self.fns)