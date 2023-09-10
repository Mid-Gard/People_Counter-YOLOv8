import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from tkinter import ImageTk
import requests

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the YOLO model
models_dir = Path('models')
models_dir.mkdir(exist_ok=True)
SEG_MODEL_NAME = "yolov8n-seg"
seg_model = YOLO(models_dir / f'{SEG_MODEL_NAME}.pt').to(device)

# Create the main window
root = tk.Tk()
root.title("Object Detection - Saro Farm")

# Maximize the window
root.state('zoomed')

# Create a border between the two columns
separator = ttk.Separator(root, orient="vertical")
separator.grid(row=0, column=0, sticky="ns")

# Configure the main window background color
root.configure(bg="lightgray")

# Column 1: Divide into two rows
frame_column1 = tk.Frame(root, padx=20, pady=20, bg="#f0f0f0")
frame_column1.grid(row=0, column=0, sticky="nsew")

# Row 1: Dropdown, Buttons
frame_row1 = tk.Frame(frame_column1, bg="#f0f0f0")
frame_row1.grid(row=0, column=0, sticky="nsew")

# Create a label for displaying object detection results
result_label = ttk.Label(frame_row1, text="", background="#f0f0f0", font=("Helvetica", 12))
result_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

# Row 2: Reserved Space with border
frame_row2 = tk.Frame(frame_column1, bg="#f0f0f0")
frame_row2.grid(row=1, column=0, sticky="nsew")

# Reserved Space with rounded border
video_label = ttk.Label(frame_row2, text="", font=("Helvetica", 14, "bold"), relief="ridge", borderwidth=2)
video_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

# Column 2: Video
frame_column2 = tk.Frame(root, bg="black")
frame_column2.grid(row=0, column=1, sticky="nsew")

# Function to perform object detection on a frame and update the GUI
def perform_object_detection():
    global frame

    # Get the frame from the stream
    frame = get_frame_from_stream(STREAM_URL)
    frame = cv2.resize(frame, (640, 640))

    if frame is None:
        print("Failed to retrieve frame from the stream.")
        return

    # Convert BGR to RGB channel order
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Normalize the frame to have pixel values in the range [0, 1]
    frame = frame / 255.0

    # Move the frame data to the GPU
    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(device)

    # Perform inference on the frame
    res = seg_model(frame_tensor)

    # Initialize box_multi_list for this frame
    box_multi_list = []

    # ... (your object detection code) ...

    # Convert the OpenCV frame to a format that can be displayed in tkinter
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    frame_tk = ImageTk.PhotoImage(frame_pil)

    # Update the video_label in the GUI with the video feed
    video_label.config(image=frame_tk)
    video_label.image = frame_tk

    # Schedule the next object detection in the GUI (e.g., after 100 milliseconds)
    root.after(100, perform_object_detection)

# Function to get a frame from the video stream
def get_frame_from_stream(url: str) -> np.ndarray:
    response = requests.get(url)
    if response.status_code == 200:
        frame = np.asarray(bytearray(response.content), dtype=np.uint8)
        return cv2.imdecode(frame, cv2.IMREAD_COLOR)
    else:
        return None

# Replace the STREAM_URL with the URL of the video stream
STREAM_URL = "http://192.168.29.187:8080/shot.jpg"

# Start the object detection loop in the background
perform_object_detection()

# Start the GUI main loop
root.mainloop()
