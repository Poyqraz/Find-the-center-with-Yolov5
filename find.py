import cv2
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import math
from math import sqrt
import datetime
import imutils
from yolov5 import YOLOv5

# Dosya yolu
cap = cv2.VideoCapture(0)

# Sınıf isimleri
classesFile = "classes2.txt"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Model yapılandırması ve ağırlıkları
modelConfiguration = 'data.yaml'
modelWeights = 'best.pt'

# YOLOv5 modelini yükleme (Yolov8 henüz mevcut değil)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True, trust_repo=True)
#model = torch.hub.load('ultralytics/yolov5', 'custom', path=modelWeights, force_reload=True, trust_repo=True)


# Nesne tespiti için gerekli parametreler
confThreshold = 0.5
nmsThreshold = 0.3

def findObjects(results, img):
    hT, wT, cT = img.shape
    bbox_coordinates = []
    for detection in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls_id = detection
        if conf > confThreshold:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (25, 50, 255), 2)
            cv2.putText(img, f'{classNames[int(cls_id)].upper()} {int(conf * 100)}%',
                        (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 50, 255), 2)
            bbox_coordinates.append((int(x1), int(y1), int(x2-x1), int(y2-y1)))
    return bbox_coordinates

while True:
    success, img = cap.read()
    if not success:
        break

    # Frame üzerinde nesne tespiti
    results = model(img)

    # Tespit edilen nesneleri gösterme
    bbox_list = findObjects(results, img)
    print(bbox_list)

    for bbox in bbox_list:
        x, y, w, h = bbox
        wT, hT, _ = img.shape
        horizontal_difference = x + w // 2 - wT // 2
        vertical_difference = y + h // 2 - hT // 2

        if horizontal_difference > 0:
            cv2.putText(img, "Right:" + ('%.2f' % float(horizontal_difference)), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
        elif horizontal_difference < 0:
            cv2.putText(img, "Left:" + ('%.2f' % float(-horizontal_difference)), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

        if vertical_difference > 0:
            cv2.putText(img, "Down:" + ('%.2f' % float(vertical_difference)), (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
        elif vertical_difference < 0:
            cv2.putText(img, "Up:" + ('%.2f' % float(-vertical_difference)), (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

        # Merkez noktaları çizme
        cv2.circle(img, (wT // 2, hT // 2), 3, (0, 0, 0), 2)
        cv2.circle(img, (x + w // 2, y + h // 2), 3, (0, 0, 255), 2)


    # Görüntüyü gösterme
    cv2.imshow('YOLOv5 Object Detection', img)

    # 'q' tuşuna basıldığında döngüden çıkma
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırakma
cap.release()
cv2.destroyAllWindows()
