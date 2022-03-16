import pickle
import torch
import torch.nn as nn
import time
import detect_model
import pytorchtools

# load tensor dataset
with open("detect_train_data.pickle","rb") as fr:
    train = pickle.load(fr)
with open("detect_label_data.pickle","rb") as fr:
    label = pickle.load(fr)

with open("detect_test_data.pickle","rb") as fr:
    test = pickle.load(fr)
with open("detect_tlabel_data.pickle","rb") as fr:
    tlabel = pickle.load(fr)

# set batch tensor 
train_dataset = []
batch_size = 2

for i in range(0,len(train),batch_size):
    train_temp = torch.cat([train[i],train[i+1]])
    label_temp = torch.cat([label[i],label[i+1]])
    train_dataset.append([train_temp,label_temp])

test_dataset = []
for i in range(0,len(test),batch_size):
    test_temp = torch.cat([test[i],test[i+1]])
    tlabel_temp = torch.cat([tlabel[i],tlabel[i+1]])
    test_dataset.append([test_temp,tlabel_temp])

# set train parameters
img_size = 128
in_dim = 3
out_dim = 1
num_filters = 64
num_epoch = 300
lr = 0.001

size = len(train_dataset)
tsize = len(test)
tnum_batches = len(test_dataset)

# instantiate the VGG class as defined, put it on device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = detect_model.Detector(in_dim=in_dim,out_dim=out_dim,num_filter=num_filters).to(device)

# set loss function, optimizer
#loss_func = nn.CrossEntropyLoss()
loss_func = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

val_loss = 1
early_stopping = 0
for i in range(num_epoch+1):
    start_time = time.time()
    for j in range(len(train_dataset)):
        X = train_dataset[j][0]
        y = train_dataset[j][1]
        
        optimizer.zero_grad()
        output = model.forward(X)
        
        loss = loss_func(output,y)
        loss.backward()
        
        optimizer.step()
    
    #######test###########
    test_loss, correct = 0, 0
    with torch.no_grad():
        for tX, ty in test_dataset:
            pred = model(tX)
            test_loss += loss_func(pred, ty).item()
            # correct_prediction = torch.argmax(pred, 1) == ty
            # correct += correct_prediction.type(torch.float).sum().item()
            predict = torch.argmax(pred.long(), 1) + 1
            target = ty.long() + 1
            correct += torch.sum((predict == target) * (target > 0)).item()

    test_loss /= 200
    correct /= 200
    end_time = time.time()
    running_time = end_time - start_time

    if (i+1) % 5 ==0:
        print(f"epoch:{(i+1):4d}  ||  train Loss:{loss.item():.6f}  ||  test Loss:{test_loss:.6f}  ||  acc:{correct*100}% || time:{running_time*5:.3f}")
        if test_loss < val_loss:
            print(f'loss advanced : {val_loss:.6f} --> {test_loss:.6f}___model saved___!!!!!!')
            val_loss = test_loss
            torch.save(model.state_dict(), f'./weights/detect_1000img_300epoch.pth')

        else: # apply early stopping
            early_stopping+=1
            if early_stopping >= 20:
                break

print("Done!")


