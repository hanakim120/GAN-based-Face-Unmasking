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
        
        self.cfg = cfg
        self.img_binary_folder = cfg.img_binary_folder
        self.img_gt_folder = cfg.img_gt_folder
        self.load_images()
        
    def load_images(self):
        self.fns = []

        img_gt_paths = sorted(os.listdir(self.img_gt_folder))
        img_binary_paths = sorted(os.listdir(self.img_binary_folder))

        for img_gt_name, img_binary_name in zip(img_gt_paths,img_binary_paths):      
            img_gt_path = os.path.join(self.img_gt_folder, img_gt_name)
            img_binary_path = os.path.join(self.img_binary_folder, img_binary_name)
            if os.path.isfile(img_binary_path): 
                self.fns.append([img_gt_path, img_binary_path])

    def __getitem__(self, index):
        img_gt_path, img_binary_path = self.fns[index]
        img_gt = cv2.imread(img_gt_path)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        img_gt = cv2.resize(img_gt, (self.cfg.img_size_h, self.cfg.img_size_w))
        img_binary = cv2.imread(img_binary_path, 0)
        img_binary[img_binary>0]=1.0
        img_binary = np.expand_dims(img_binary, axis=0)
    
        img_gt = torch.from_numpy(img_gt.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        img_binary = torch.from_numpy(img_binary.astype(np.float32)).contiguous()
        
        return img_gt, img_binary
    
    def collate_fn(self, batch):
        imgs = torch.stack([i[0] for i in batch])
        masks = torch.stack([i[1] for i in batch])
        return {
            'imgs': imgs,
            'masks': masks
        }
    
    def __len__(self):
        return len(self.fns)
