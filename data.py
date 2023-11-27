from tqdm import tqdm
import numpy as np
import cv2
import os
import random
import math
import torch


def dct(img, n):
    '''
    3channel 2d DCT algorithm
    img: Array, image data
    n:int, decide the ratio of the high-pass filtering
    '''
    tempdata = np.zeros_like(img)

    for m in range(3):
        img_dct = cv2.dct(np.array(img[:, :, m], np.float32))
        k = min(list(img_dct.shape))//10
        for i in range(k*(n+1)):
            for j in range(k*(n+1)):
                if i+j <= k*(n+1):
                    img_dct[i, j] = 0
        img_idct = np.array(cv2.idct(img_dct), np.uint8)
        # cv2.imshow("",img_dct)
        # cv2.waitKey(0)

        tempdata[:, :, m] = img_idct

    return tempdata

def get_data(allreal, allfake, batch_size, device, path):

    c = 0

    # print(allreal)
    # print(allreal)

    # ------------------------------------
    # 이미지 불러오고 데이터셋 정렬해줘야됨 이거 학습을 어떻게 시키는지 몰라도 데이터셋 구성 필요할듯
    data_mx=min(len(allreal), len(allfake))
    print(data_mx)
    allreal=allreal[:data_mx]
    allfake=allfake[:data_mx]
    print(f'RealData:{len(allreal)},FakeData:{len(allfake)}')
    print()
    random.shuffle(allreal)
    random.shuffle(allfake)
    while True:

        allrealtemp = allreal[c*batch_size:c*batch_size+batch_size]
        allfaketemp = allfake[c*batch_size:c*batch_size+batch_size]
        if len(allfaketemp) != 0 or len(allrealtemp) != 0:
            data1 = []
            data2 = []
            data3 = []
            data4 = []
            imgs = []

            label = []
            for i in allrealtemp:

                img0 = np.array(cv2.imread(path[0]+'/'+i))
                for img in [img0, cv2.GaussianBlur(img0, (3, 3), 0), cv2.GaussianBlur(img0, (5, 5), 0)]:
                    img = cv2.resize(img, (299, 299))

                    newimg = np.zeros_like(img)

                    for m in range(3):
                        imgx = cv2.resize(img[:, :, m], (299, 299))
                        f = np.fft.fft2(imgx)
                        fshift = np.fft.fftshift(f)
                        res = np.log(np.abs(fshift))

                        newimg[:, :, m] = res
                    data1.append(np.transpose(
                        cv2.resize(img, (299, 299)), [2, 0, 1]))
                    data2.append(np.transpose(cv2.resize(
                        dct(img, 4), (299, 299)), [2, 0, 1]))
                    data3.append(np.transpose(cv2.resize(
                        dct(img, 5), (299, 299)), [2, 0, 1]))
                    data4.append(np.transpose(
                        cv2.resize(newimg, (299, 299)), [2, 0, 1]))
                    imgs.append(img)

                    label.append(0)

            for i in allfaketemp:
                img0 = np.array(cv2.imread(path[1]+'/'+i))
                for img in [img0, cv2.GaussianBlur(img0, (3, 3), 0), cv2.GaussianBlur(img0, (5, 5), 0)]:
                    img = cv2.resize(img, (299, 299))
                    newimg = np.zeros_like(img)

                    for m in range(3):
                        imgx = cv2.resize(img[:, :, m], (299, 299))
                        f = np.fft.fft2(imgx)
                        fshift = np.fft.fftshift(f)
                        res = np.log(np.abs(fshift))

                        newimg[:, :, m] = res
                    data1.append(np.transpose(
                        cv2.resize(img, (299, 299)), [2, 0, 1]))
                    data2.append(np.transpose(cv2.resize(
                        dct(img, 4), (299, 299)), [2, 0, 1]))
                    data3.append(np.transpose(cv2.resize(
                        dct(img, 5), (299, 299)), [2, 0, 1]))
                    data4.append(np.transpose(
                        cv2.resize(newimg, (299, 299)), [2, 0, 1]))
                    imgs.append(img)

                    label.append(1)

            X1 = torch.tensor(np.array(data1, dtype=np.float32),
                              device=device).float()
            X2 = torch.tensor(np.array(data2, dtype=np.float32),
                              device=device).float()
            X3 = torch.tensor(np.array(data3, dtype=np.float32),
                              device=device).float()
            X4 = torch.tensor(np.array(data4, dtype=np.float32),
                              device=device).float()

            y = torch.tensor(np.array(label, dtype=np.float32),
                             device=device).long()
            c += 1
            yield X1, X2, X3, X4, y, imgs
        else:
            break