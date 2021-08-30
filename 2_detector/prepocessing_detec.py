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
    
    '''
    ###############################참고용###############################
    import os
import cv2
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from models import networks
    import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image
from torchvision import models

model1 = Net()
model1.load_state_dict(torch.load('./model_13_3100.pth'))
img_folder = './ourm'
mask_folder = './ourb'
img_paths = os.listdir(img_folder)
mask_paths = os.listdir(mask_folder)

for img_name,mask_name in zip(img_paths,mask_paths):      
    img_path = os.path.join(img_folder, img_name)
    mask_path = os.path.join(mask_folder, mask_name)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512,512))
    mask = cv2.imread(mask_path, 0)
    mask[mask>0]=1.0
    mask = np.expand_dims(mask, axis=0)

    img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
    mask = torch.from_numpy(mask.astype(np.float32)).contiguous()
    '''
