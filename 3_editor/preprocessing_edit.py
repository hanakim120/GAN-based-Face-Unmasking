import os
import torch
import torch.nn as nn
import torch.utils.data as data
import cv2
import numpy as np
from tqdm import tqdm

class FacemaskDataset(data.Dataset):
    def __init__(self, cfg):
        self.root_dir = cfg.root_dir
        #cfg.root_dir = "./datasets/GAN/"
        
        self.cfg = cfg
        self.mask_folder = os.path.join(self.root_dir, '6_sharpen')
        self.img_folder = os.path.join(self.root_dir, '3_masked')
        self.load_images()
        
    def load_images(self):
        self.fns = []
        idx = 0
        img_paths = sorted(os.listdir(self.img_folder))
        mask_paths = sorted(os.listdir(self.mask_folder))
        #self.img_folder = "./datasets/GAN/6_sharpen"
        #000150-with-mask_black
        #000124-with-mask_blue
        #000134-with-mask
        for img_name,mask_name in zip(img_paths,mask_paths):      
            img_path = os.path.join(self.img_folder, img_name)
            mask_path = os.path.join(self.mask_folder, mask_name)
            if os.path.isfile(mask_path): 
                self.fns.append([img_path, mask_path])

    def __getitem__(self, index):
        img_path, mask_path = self.fns[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.cfg.img_size_h, self.cfg.img_size_w))
        
        
        mask = cv2.imread(mask_path, 0)
        
        mask[mask>0]=1.0
        mask = np.expand_dims(mask, axis=0)
    
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32)).contiguous()
        return img, mask
    
    def collate_fn(self, batch):
        imgs = torch.stack([i[0] for i in batch])
        masks = torch.stack([i[1] for i in batch])
        return {
            'imgs': imgs,
            'masks': masks
        }
    
    def __len__(self):
        return len(self.fns)