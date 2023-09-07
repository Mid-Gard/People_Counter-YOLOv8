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
# STREAM_URL = "http://192.168.137.156:8080/video?type=some.mjpeg"
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

    # result_pil_image = Image.fromarray(res[0].plot()[:, :, ::-1])

    # # Convert the PIL image to an array to use with OpenCV
    # result_cv_image = np.array(result_pil_image)

    # # Display the result in the OpenCV window
    # cv2.imshow("Result Image", result_cv_image)

    # # Wait for 1 second (1000 milliseconds) before displaying the next frame
    # cv2.waitKey(500)

        # --------- list that stores the centroids of the current frame---------#
    centr_pt_cur_fr = []

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

    # ----------- bounding boxes for person detections---------------#
    bbox = []
    for i in idx:
        temp = bboxes[i]
        # print("this is temp", temp)
        bbox.append(temp)

        # Convert to bbox to multidimensional list
        box_multi_list = [arr.tolist() for arr in bbox]
        # print("this are final human detected boxes")
        # print(box_multi_list)

    # ------------ drawing of bounding boxes-------------#
    for box in box_multi_list:
        (x, y, x2, y2) = box

        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
        cx = int((x+x2)/2)
        cy = int((y+y2)/2)
        centr_pt_cur_fr.append((cx, cy))
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    # print("this are the centroids in the current frame")
    # print(centr_pt_cur_fr)

    # ------------- counting of total people in the footage ------------#
    head_count = len(centr_pt_cur_fr)

    # counting the number of faces with count_var variable
    count_var = head_count

    # displaying the face count on the screen for experiment purpose
    # cv2.putText(frame, f'{head_count}', (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    text = f'Head Count: {head_count}'
    cv2.putText(frame, text, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow("Result Image", frame)
    # Check if the user presses the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
