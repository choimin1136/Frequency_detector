import torch
from torch import nn
import cv2
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import os
# import dlib
from face_detector import YoloDetector
from model import XCE4_Net
from data import dct
from tqdm import tqdm
from os.path import join
import math

model = XCE4_Net()

model.load_state_dict(torch.load("train_model\\best_0_detector_1.000000.pth"))# 见 release
model.eval()

model.cuda()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Text variables
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
THICKNESS = 2
FONT_SCALE = 1

def plot_result(video_path):

    reader = cv2.VideoCapture(video_path)
    output_path = "./"
    START_FRAME = 0
    END_FRAME = None

    video_fn = video_path.split('/')[-1].split('.')[0]+'.avi'
    os.makedirs(output_path, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None

    # hog_face_detetor = dlib.get_frontal_face_detector()
    face_detector = YoloDetector()

    # Frame numbers and length of output video
    frame_num = 0
    assert START_FRAME < num_frames - 1
    END_FRAME = END_FRAME if END_FRAME else num_frames
    pbar = tqdm(total=END_FRAME-START_FRAME)

    avg_l=[]

    while reader.isOpened():
        _, img = reader.read()
        if img is None:
            break
        # cv2.imshow('img',img)
        origin=img

        # img = cv2.imread(image)
        # print(image.shape)
        imgg3 = cv2.GaussianBlur(img,(3,3),0)
        imgg5 = cv2.GaussianBlur(img,(5,5),0)

        cv2.imwrite("origin.png", img)
        cv2.imwrite("gaussian3x3.png", imgg3)
        cv2.imwrite("gaussian5x5.png", imgg5)

        height, width = img.shape[:2]


        frame_num += 1

        if frame_num < START_FRAME:
            continue
        pbar.update(1)

        # Image size
        height, width = img.shape[:2]

        # Init output writer
        if writer is None:
            writer = cv2.VideoWriter(join(output_path, video_fn), fourcc, fps,
                                     (height, width)[::-1])

        # 2. Detect with yolo
        bboxes,points=face_detector.predict(img)
        # detections = hog_face_detetor(img,1)

        data1 = []
        data2 = []
        data3 = []
        imgs = []

        # ---------------------------------
        box=None
        if len(bboxes[0]):
            box=bboxes[0]
            face=bboxes[0][0]
            # x, y, size = get_boundingbox(box, width, height)

            # for face in box:
                # x = face.left()
                # y = face.top()
                # r = face.right()
                # b = face.bottom()
            x = face[0]
            y = face[1]
            r = face[2]
            b = face[3]

            face_rect = img[np.maximum(y-10,0):b+10, np.maximum(x-10,0):r+10]


            img = cv2.resize(face_rect,(299,299))
            newimg = np.zeros_like(img)

            

            for m in range(3):
                imgx = cv2.resize(img[:,:,m],(299,299))
                f = np.fft.fft2(imgx)
                fshift = np.fft.fftshift(f)
                res = np.log(np.abs(fshift))

                newimg[:,:,m] = res
            f = np.fft.fft2(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            fshift = np.fft.fftshift(f)
            res = np.log(np.abs(fshift))

            res = cv2.normalize(res, None, 0, 1, cv2.NORM_MINMAX)
            # plt.subplot(1,5,2), plt.imshow(np.array(newimg, np.uint8))
            # plt.axis('off')
            # plt.show()
            data1 = torch.tensor(np.array([np.transpose(cv2.resize(np.array(img, dtype=np.float64),(299,299)), [2,0,1])], dtype=np.float64),device=device).float()
            data2 = torch.tensor(np.array([np.transpose(cv2.resize(np.array(dct(img,4), dtype=np.float64),(299,299)), [2,0,1])], dtype=np.float64),device=device).float()
            data3 = torch.tensor(np.array([np.transpose(cv2.resize(np.array(dct(img,5), dtype=np.float64),(299,299)), [2,0,1])], dtype=np.float64),device=device).float()
            data4 = torch.tensor(np.array([np.transpose(cv2.resize(np.array(newimg, dtype=np.float64),(299,299)), [2,0,1])], dtype=np.float64),device=device).float()

            # plt.subplot(1,5,1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),plt.title(str(nn.Softmax(dim=1)(model(data1, data2, data3,data4))[0][1].detach().cpu().numpy()))
            # plt.axis('off')
            # plt.show()
            # for i in range(3):

            #     plt.subplot(1,5,i+3), plt.imshow(cv2.resize(dct(img,i+3),(299,299)))
            #     plt.axis('off')
            # plt.show()
            avg_l.append(round(nn.Softmax(dim=1)(model(data1, data2, data3,data4))[0][1].item(),3))
            avg=round(sum(avg_l)/len(avg_l),3)
            color=(0,255,0) if avg <=0.5 else (0,0,255)
            text='real:' if avg <=0.5 else 'fake:'

            cv2.putText(origin,str(nn.Softmax(dim=1)(model(data1, data2, data3,data4))[0][0].item()),(0,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
            cv2.putText(origin,str(nn.Softmax(dim=1)(model(data1, data2, data3,data4))[0][1].item()),(0,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

            cv2.rectangle(origin,(np.maximum(x-10,0),np.maximum(y-10,0)),(np.maximum(x+135,0),np.maximum(y+20,0)),color=color,thickness=-1)
            cv2.putText(origin,text+str(avg),(np.maximum(x-5,0),np.maximum(y+15,0)),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),1)
            cv2.imshow('a',res)
            cv2.rectangle(origin,(np.maximum(x-10,0),np.maximum(y-10,0)),(r+10,b+10),color=color)
        
            cv2.imshow('origin',origin)
            cv2.waitKey(33)
 
        else:
            continue
    pbar.close()

    plt.savefig("output6.png")
if __name__ == "__main__":
    # plot_result("2.png")
    plot_result("videos/c.mp4")