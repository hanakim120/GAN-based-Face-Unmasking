import pickle
import torch
import torch.nn as nn

import detect_model
from metrics import dicecoeff, pixelacc
from torch.utils.tensorboard import SummaryWriter

summary = SummaryWriter()
# https://tutorials.pytorch.kr/beginner/basics/optimization_tutorial.html 
#tensor dataset 불러오기
with open("detect_train_data.pickle","rb") as fr:
    train = pickle.load(fr)
with open("detect_label_data.pickle","rb") as fr:
    label = pickle.load(fr)

with open("detect_test_data.pickle","rb") as fr:
    test = pickle.load(fr)
with open("detect_tlabel_data.pickle","rb") as fr:
    tlabel = pickle.load(fr)

########batch tensor 설정########
#가지고있는것
#train, label
#test, tlabel
#순서
#1. img를 하나씩 넣는 [x] 준비
#2. [x]들을 가지고 있는 img개수가 batch_size가 되면 cat 후 train_dataset에 넣음

train_size=len(train)
train_dataset=[]
#batch,concat용 리스트
train_batch=[train[0]]
label_batch=[label[0]]
batch_size = 4
if batch_size==1:
    train_temp = torch.cat(train_batch)
    label_temp = torch.cat(label_batch)
    train_dataset.append([train_temp,label_temp])
    train_batch, label_batch = [],[]
for i in range(1,train_size):
    if (i+1)%batch_size == 0:
        train_batch.append(train[i])
        label_batch.append(label[i])
        
        train_temp = torch.cat(train_batch)
        label_temp = torch.cat(label_batch)
        train_dataset.append([train_temp,label_temp])
        train_batch, label_batch = [],[]
    else:
        train_batch.append(train[i])
        label_batch.append(label[i])
    

test_size=len(test)
test_dataset=[]
#batch,concat용 리스트
test_batch=[test[0]]
tlabel_batch=[tlabel[0]]
test_batch_size = 1
if test_batch_size==1:
    test_temp = torch.cat(test_batch)
    tlabel_temp = torch.cat(tlabel_batch)
    test_dataset.append([test_temp,tlabel_temp])
    test_batch, tlabel_batch = [],[]
for i in range(1,test_size):
    if (i+1)%test_batch_size == 0:
        test_batch.append(test[i])
        tlabel_batch.append(tlabel[i])
        
        test_temp = torch.cat(test_batch)
        tlabel_temp = torch.cat(tlabel_batch)
        test_dataset.append([test_temp,tlabel_temp])
        test_batch, tlabel_batch = [],[]
    else:
        test_batch.append(test[i])
        tlabel_batch.append(tlabel[i])


#학습 파라미터 설정
#img_size = 128
in_dim = 3
out_dim = 1
num_filters = 64
num_epoch = 300
lr = 0.001

patience=30
flag = 0


# 앞서 정의한대로 vGG 클래스를 인스턴스화 하고 지정한 장치에 올립니다.
device = torch.device("cuda")
model = detect_model.Detector(in_dim=in_dim,out_dim=out_dim,num_filter=num_filters).to(device)

# 손실함수 및 최적화함수를 설정합니다.
#loss_func = nn.CrossEntropyLoss()
loss_func = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

import time
val_loss = 1
early_stopping = 0
check = 1
for i in range(num_epoch+1):
    start_time = time.time()
    #######train###########
    for j in range(len(train_dataset)):
        X = train_dataset[j][0]
        y = train_dataset[j][1]
        
        optimizer.zero_grad()
        output = model.forward(X)
        
        loss = loss_func(output,y)
        loss.backward()
        
        optimizer.step()
        if check<10:
            print("Goooooood!")
            check+=1
    
    #######test###########
    metrics = [pixelacc.PixelAccuracy(1)]
    test_loss, correct = 0, 0
    with torch.no_grad():
        for tX, ty in test_dataset:
            pred = model(tX)
            test_loss += loss_func(pred, ty).item()
            for metric in metrics:
                metric.update(pred, ty)
                

    test_loss /= len(test_dataset)
    end_time = time.time()
    running_time = end_time - start_time

    summary.add_scalar('loss', loss.item(), i+1)
    summary.add_scalar('test_loss', test_loss, i+1)
    
    print(f"epoch:{(i+1):4d}  ||  train Loss:{loss.item():.6f}  ||  test Loss:{test_loss:.6f}  ||  time:{running_time*5:.3f}")
    for metric in metrics:
        print(metric)
    if (i+1) % 3 ==0:    
        if test_loss < val_loss:
            print(f'loss advanced : {val_loss:.6f} --> {test_loss:.6f}___model saved___!!!!!!')
            val_loss = test_loss
            torch.save(model.state_dict(), f'/home/ubuntu/GAN-based-Face-Unmasking/detector/weights/detect_{i+1}epochs.pth')
        else: #loss가 좋아지지 않는다면
            early_stopping+=1
            if early_stopping >= 20:
                break

summary.close()
print("Done!")
