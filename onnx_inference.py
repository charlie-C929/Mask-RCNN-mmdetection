import os

import numpy as np
import onnxruntime as ort

from mmcv.ops import get_onnxruntime_op_path

import time

import cv2
from sklearn.metrics import jaccard_score

import torch

from torchvision.ops.boxes import nms


def preprocess(img):
    norm_mean = (0.5, 0.5, 0.5)
    norm_std = (0.5, 0.5, 0.5)
    data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = data.astype(np.float32) / 255.0
    data = (data - norm_mean) / norm_std
    data = np.transpose(data, (2, 0, 1)).astype(np.float32)
    data = np.expand_dims(data, axis=0)
    return data


ort_custom_op_path = get_onnxruntime_op_path()
assert os.path.exists(ort_custom_op_path)
session_options = ort.SessionOptions()
session_options.register_custom_ops_library(ort_custom_op_path)
## exported ONNX model with custom operators
onnx_file = 'checkpoints/result.onnx'


#time.sleep(10)
sess = ort.InferenceSession(onnx_file, session_options)
start = time.time()
img = cv2.imread("/home/yi/Yi/mmdetection/Mask-RCNN-mmdetection/data/coco/000000017627.jpg")
input_data = preprocess(img)
onnx_results = sess.run(None, {'input' : input_data})
end=time.time()
print("Inference time:",end-start,"s")
bboxes = onnx_results[0][0]
labels = onnx_results[1][0]
masks = onnx_results[2][0]

needindexs = np.where(bboxes[:,-1] > 0.2)
bboxes = bboxes[needindexs]
labels = labels[needindexs]
masks = masks[needindexs]

boxs = []
score = []

for index,box in enumerate(bboxes):
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])
    score.append(float(box[4]))
    boxs.append([x1,y1,x2,y2])

boxs = torch.Tensor(boxs)
score = torch.Tensor(score)
needindexs = nms(boxs,score,0.3)

boxes = boxs[needindexs]
labels = labels[needindexs]
masks = masks[needindexs]

for index,box in enumerate(boxes):
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])
    
    mask = masks[index]
    print(mask.shape)
    mask = mask * 255
    ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
 
    black = np.zeros_like(img)
    black = cv2.cvtColor(black, cv2.COLOR_BGR2GRAY)
    black = mask.astype(np.uint8)
 
    contours, hierarchy = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 1:
        contours = np.array(contours).reshape((-1, 1, 2))
        cv2.polylines(img, [contours], isClosed=True, color=(0, 0, 255), thickness=3)
    else:
        for contour in contours:
            contour = np.array(contour).reshape((-1, 1, 2))
            cv2.polylines(img, [contour], isClosed=True, color=(0, 0, 255), thickness=3)
   

    alpha=0.8
    mask = mask.astype(bool)
    random_colors = np.array([0,255,255])
    img[mask] = img[mask] * (1 - alpha) + random_colors * alpha

    # cv2.imshow('',black)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imshow('',img)
cv2.waitKey(0)

