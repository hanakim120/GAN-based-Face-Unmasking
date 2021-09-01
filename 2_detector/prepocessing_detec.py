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
        self.root_dir = cfg.root_dir
        self.cfg = cfg
        self.train = train

        self.img_binary_folder = cfg.img_binary_folder
        self.img_masked_folder = cfg.img_masked_folder

        # detector validation 사용할지는 회의에서 논의해보기
        # if self.train:
        #     self.df = pd.read_csv(cfg.train_anns)
        # else:
        #     self.df = pd.read_csv(cfg.val_anns)

        self.load_images()        
        
    def load_images(self):
        self.fns = []

        img_binany_paths = os.listdir(self.img_binary_folder)
        img_masked_paths = os.listdir(img_masked_folder)

       for img_binary_name, img_masked_name in zip(img_binany_paths, img_masked_paths) :
            img_binary_path = os.path.join(self.img_binary_folder, img_binary_name)
            img_masked_path = os.path.join(self.img_masked_folder, img_masked_name)
            if os.path.isfile(img_binary_path) :
                self.fns.append([img_masked_path, img_binary_path])



    def __getitem__(self, index) :
        img_masked_path, img_binary_path = self.fns[index]
        img_masked = cv2.imread(img_masked_path)
        img_masked = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_masked = cv2.resize(img, (self.cfg.img_size, self.cfg.img_size))
        img_binary = cv2.imread(img_binary_path, 0)
        img_binary[img_binary>0]=1.0
        img_binary = np.expand_dims(img_binary, axis=0)

        img_masked = torch.from_numpy(img_masked.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        img_binary = torch.from_numpy(img_binary.astype(np.float32)).contiguous()

        return img_masked, img_binary


    def collate_fn(self, batch):
        imgs = torch.stack([i[0] for i in batch])
        masks = torch.stack([i[1] for i in batch])
        return {
            'imgs': imgs,
            'masks': masks
        }
    
    def __len__(self):
        return len(self.fns)
