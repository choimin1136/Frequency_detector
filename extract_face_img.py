from tqdm import tqdm
import numpy as np
import cv2
import os
import random
# import dlib

from face_detector import YoloDetector
REAL_SOURCE_DATA = "D:/Dataset/real"
FAKE_SOURCE_DATA = "D:/Dataset/fake"
REAL_OUTPUT_DATA = "D:/Dataset/ext_face/real"
FAKE_OUTPUT_DATA = "D:/Dataset/ext_face/fake"

TEST_REAL_SOURCE_DATA='test/videos/real'
TEST_FAKE_SOURCE_DATA='test/videos/fake'
TEST_REAL_OUTPUT_DATA='test/real'
TEST_FAKE_OUTPUT_DATA='test/fake'

def extract(source_data, output_data):
    c = 0

    # hog_face_detetor = dlib.get_frontal_face_detector()
    face_detector = YoloDetector()

    # for i in tqdm(os.listdir(source_data)):
    for i in os.listdir(source_data):
        print(source_data+'/'+i)
        START_FRAME = 0
        END_FRAME = None

        reader = cv2.VideoCapture(source_data+'/'+i)
        # img = cv2.imread(source_data+"/"+i)
        fps = reader.get(cv2.CAP_PROP_FPS)
        num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_num = 0
        assert START_FRAME < num_frames - 1
        END_FRAME = END_FRAME if END_FRAME else num_frames
        pbar = tqdm(total=END_FRAME-START_FRAME)
        
        while reader.isOpened():
            _, img = reader.read()
            # print(img.shape)
            
            
            frame_num += 1
            if frame_num > END_FRAME-200:
                break

            if frame_num < START_FRAME:
                continue
            pbar.update(1)

            # boxes=[[]]
            try:
                boxes,point = face_detector(img)
            except:
                break

            # try:
            if len(boxes[0]):
                face = boxes[0][0]
                x = face[0]
                y = face[1]
                r = face[2]
                b = face[3]

                face_rect = img[np.maximum(y-10, 0):b+10, np.maximum(x-10, 0):r+10]
                if face_rect is not None:
                    imgf = cv2.resize(face_rect, (299, 299))
                    cv2.imwrite(output_data+"/"+'{0:09d}'.format(c)+'.jpg', imgf)
                    c += 1
            cv2.waitKey(33)
            # except:
            #     continue
        pbar.close()

if __name__ == "__main__":
    # real_list = os.listdir(REAL_SOURCE_DATA)
    # fake_list = os.listdir(FAKE_SOURCE_DATA)

    # extract(REAL_SOURCE_DATA, REAL_OUTPUT_DATA)
    extract(FAKE_SOURCE_DATA, FAKE_OUTPUT_DATA)
    # extract(TEST_REAL_SOURCE_DATA, TEST_REAL_OUTPUT_DATA)
    # extract(TEST_FAKE_SOURCE_DATA, TEST_FAKE_OUTPUT_DATA)