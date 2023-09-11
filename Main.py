import torch
import requests

# from ultralytics.yolo.utils.plotting 

import ultralytics.models.yolo
import ultralytics.utils
from PIL import Image
import numpy as np
from typing import Tuple, Dict
from ultralytics import YOLO
import os
from pathlib import Path
import cv2
# Fetch the notebook utils script from the openvino_notebooks repo
import urllib.request
from random import random

# urllib.request.urlretrieve(
#     url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py',
#     filename='../openvino_notebooks/notebooks/230-yolov8-optimization/notebook_utils.py'
# )

models_dir = Path('models')
models_dir.mkdir(exist_ok=True)


def get_frame_from_stream(url: str) -> np.ndarray:
    response = requests.get(url)
    if response.status_code == 200:
        frame = np.asarray(bytearray(response.content), dtype=np.uint8)
        return cv2.imdecode(frame, cv2.IMREAD_COLOR)
    else:
        return None


# Replace the IMAGE_PATH with the URL of the video stream
STREAM_URL = "http://192.168.29.187:8080/shot.jpg"
# STREAM_URL = "http://192.168.29.187:8080/video?type=some.mjpeg"
SEG_MODEL_NAME = "yolov8n-seg"
seg_model = YOLO(models_dir / f'{SEG_MODEL_NAME}.pt')

while True:
    # Get the frame from the stream
    frame = get_frame_from_stream(STREAM_URL)
    frame = cv2.resize(frame, (1200, 600))

    if frame is None:
        print("Failed to retrieve frame from the stream.")
        break

    # Perform inference on the frame
    res = seg_model(frame)

    # results = seg_model(frame)
    result = res[0]
    # ------- to get the classes of the yolo model to filter out the people---------------#
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    # print("this is classes:", classes)

    # ---------confidence level of detections-----------#
    confidence = np.array(result.boxes.conf.cpu())
    # print("this is confidence:", confidence)

    # --------- anarray of bounding boxes---------------#
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    # print("this is boxes", bboxes)

    # -------- getting indexes of the detections containing persons--------#
    idx = []
    for i in range(0, len(classes)):
        if classes[i] == 0:
            idx.append(i)

    # print("these are indexes:", idx)


    cv2.imshow("Result Image", frame)
    # Check if the user presses the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
