from tqdm import tqdm
import numpy as np
import cv2
import os
import random
import dlib

from face_detector import YoloDetector
REAL_SOURCE_DATA = "tools/reals"
FAKE_SOURCE_DATA = "fakeimgs"
REAL_OUTPUT_DATA = "tools/realsface"
FAKE_OUTPUT_DATA = "fake"

def extract(source_data, output_data):
    c = 0

    # hog_face_detetor = dlib.get_frontal_face_detector()
    face_detector = YoloDetector()

    for i in tqdm(os.listdir(source_data)):

        reader = cv2.VideoCapture(source_data+'/'+i)
        # img = cv2.imread(source_data+"/"+i)

        while reader.isOpened():
            _, img = reader.read()

            boxes,point = face_detector(img)

            try:
                if len(boxes[0][0]):
                    face = boxes[0][0]
                    x = face[0]
                    y = face[1]
                    r = face[2]
                    b = face[3]

                    face_rect = img[np.maximum(y-10, 0):b+10, np.maximum(x-10, 0):r+10]
                    if face_rect is not None:
                        imgf = cv2.resize(face_rect, (299, 299))
                        cv2.imwrite(output_data+"/"+i[:-4]+f'{c}.jpg', imgf)
                        c += 1
            except:
                continue

if __name__ == "__main__":
    extract(REAL_SOURCE_DATA, REAL_OUTPUT_DATA)
    # extract(FAKE_SOURCE_DATA, FAKE_OUTPUT_DATA)
