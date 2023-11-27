from tqdm import tqdm
import numpy as np
import cv2
import os
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import init
import matplotlib.pyplot as plt
from data import get_data
from model import XCE4_Net
import io

EPOCHS = 10
TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
TRAIN_REAL_DATAPATH = "D:/Dataset/ext_face/real"
TRAIN_FAKE_DATAPATH = "D:/Dataset/ext_face/fake"
TEST_REAL_DATAPATH = "test/real"
TEST_FAKE_DATAPATH = "test/fake"
MODEL_PATH='pretrained.pth'

def data_load(real_path, fake_path):
    real_a_data = os.listdir(real_path)
    fake_a_data = os.listdir(fake_path)
    
    max_size = min(len(real_a_data), len(fake_a_data))
    print(max_size)
    split_size = (max_size//100)*20
    print(split_size)
    
    real_data = real_a_data[split_size:max_size]
    fake_data = fake_a_data[split_size:max_size]
    real_t_data = real_a_data[:split_size]
    fake_t_data = fake_a_data[:split_size]
    
    return real_data, fake_data, real_t_data, fake_t_data


def build(model_path=None):
    model = XCE4_Net()
    # print(model)
    if model_path:
        model.load_state_dict(torch.load(model_path))
        print("Load pretrained model")
    loss_fn = nn.CrossEntropyLoss()
    from torch import optim
    # define the optimizer
    # default = SGD {lr:0.001}
    # custum1 = SGD {lr:0.1, momentum:0.9}
    # custum2 = SGD {lr:0.01, momentum:0.9}
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    return model, loss_fn, optimizer


def main(epochs):
    losses=[]
    acc=[]
    model, loss_fn, optimizer = build()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    real_data, fake_data, real_t_data, fake_t_data = data_load(TRAIN_REAL_DATAPATH,TRAIN_FAKE_DATAPATH)

    model.cuda()
    for i in range(epochs):
        print("Epoch: {}".format(i+1))
        print("Training...")
        num = 0
        best_acc = 0.0 
        for X_train1, X_train2, X_train3, X_train4, y_train, orimgs in tqdm(get_data(allreal=real_data, allfake=fake_data, batch_size=TRAIN_BATCH_SIZE, device=device, path=(TRAIN_REAL_DATAPATH,TRAIN_FAKE_DATAPATH)), unit='iters'):
            num += 1
            model.train()
            optimizer.zero_grad()

            y_pred = model(X_train1, X_train2, X_train3, X_train4)
            loss = loss_fn(y_pred, y_train)
            print(f">> loss : {loss.item()}")
            losses.append(float(loss.item()))
            loss.backward()
            optimizer.step()
            if num % 100 == 99:
                print("Testing...")
                correct = 0
                alldata = 0
                t_num = 0
                for X_test1, X_test2, X_test3, X_test4, y_test, orimgs in tqdm(get_data(allreal=real_t_data, allfake=fake_t_data, batch_size=TEST_BATCH_SIZE, device=device, path=(TRAIN_REAL_DATAPATH,TRAIN_FAKE_DATAPATH)), unit='iters'):
                    t_num += 1
                    # print(X_test1.shape)
                    alldata += X_test1.size(0)
                    model.eval()
                    with torch.no_grad():
                        y_batch_pred = model(
                            X_test1, X_test2, X_test3, X_test4)
                        index, predicted = torch.max(y_batch_pred.data, axis=1)
                        correct += predicted.eq(
                            y_test.data.view_as(predicted)).sum()
                        print(f">> loss:{losses[-1]} acc:{correct/alldata}")
                        acc.append(correct/alldata)
                    if t_num % 100 == 99:
                        break
                print("Test data acc: {}  {}/{}".format(correct /
                      alldata, correct, alldata))
                
                if correct/alldata > best_acc:
                    torch.save(model.state_dict(), "train_model/best_"+str(i) +
                               "_detector_%f.pth" % (correct/alldata))
                    best_acc = correct/alldata
                torch.save(model.state_dict(), "train_model/period_%03d_detector.pth" % i)
        with io.open(f'{epochs}_accuracy.txt', 'w', encoding='utf-8') as f:
            for item in acc:
                f.write(str(item) + '\n')
        with io.open(f'{epochs}_losses.txt', 'w', encoding='utf-8') as f:
            for item in losses:
                f.write(str(item) + '\n')
                

if __name__ == "__main__":
    main(1)