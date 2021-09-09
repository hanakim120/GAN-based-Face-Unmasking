import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy
import pickle

import torch
from torchvision import transforms
import torchvision.transforms as transforms


img_masked_name = os.listdir('/content/drive/MyDrive/Ai_project/GAN_code_review/GAN-based-Face-Unmasking/detector/image/img_masked_validate')
img_binary_name = os.listdir('/content/drive/MyDrive/Ai_project/GAN_code_review/GAN-based-Face-Unmasking/detector/image/img_binary_validate')
root = '/content/drive/MyDrive/Ai_project/GAN_code_review/GAN-based-Face-Unmasking'
print('File 개수 확인')
print(len(img_masked_name))
print(len(img_binary_name))

print('Deviece 확인')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

loader_color = transforms.Compose([transforms.ToTensor()])
loader_gray = transforms.Compose([transforms.ToTensor(),
                                  transforms.Grayscale(num_output_channels=1)])  # 토치 텐서로 변환

def image_loader_color(image_name):
    image = Image.open(image_name)
    image = image.resize((128,128))
    image = loader_color(image).unsqueeze(0)
    return image.to(device, torch.float)

def image_loader_gray(image_name):
    image = Image.open(image_name)
    image = image.resize((128,128))
    image = loader_gray(image).unsqueeze(0)
    return image.to(device, torch.float)

train = []
label = []

for i in range(len(img_masked_name)):
    image = image_loader_color(root +'/detector/image/img_masked_validate/'+ img_masked_name[i])
    train.append(image)

for i in range(len(img_binary_name)):
    image = image_loader_gray(root +'/detector/image/img_binary_validate/'+ img_binary_name[i])
    if i == 2:
      print(image.size)
    label.append(image)

#피클형태로 저장
with open("detect_test_data.pickle","wb") as fw:
    pickle.dump(train, fw)

with open("detect_tlabel_data.pickle","wb") as fw:
    pickle.dump(label, fw)

##with open("detect_train_data.pickle","wb") as fw:
##    pickle.dump(train, fw)
##
##with open("detect_label_data.pickle","wb") as fw:
##    pickle.dump(label, fw)

    
