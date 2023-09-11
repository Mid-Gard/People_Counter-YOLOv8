import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import requests
import torch
import ultralytics.models.yolo
from pathlib import Path
from ultralytics import YOLO

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the YOLO model
models_dir = Path('models')
models_dir.mkdir(exist_ok=True)
SEG_MODEL_NAME = "yolov8n-seg"
seg_model = YOLO(models_dir / f'{SEG_MODEL_NAME}.pt').to(device)

# Define stream URLs
STREAM_URLS = [
    "http://192.168.29.188:8080/shot.jpg",
    "http://192.1",
    "http://your-third-stream-url.com",
    "http://your-fourth-stream-url.com",
]

# Initialize the default stream URL
STREAM_URL = STREAM_URLS[0]

# Create a Tkinter window
root = tk.Tk()
root.title("People Counter GUI")

# Function to get a frame from the video stream
def get_frame_from_stream(url: str) -> np.ndarray:
    response = requests.get(url)
    if response.status_code == 200:
        frame = np.asarray(bytearray(response.content), dtype=np.uint8)
        return cv2.imdecode(frame, cv2.IMREAD_COLOR)
    else:
        return None

# Function to start the video feed
def start_video_feed():
    global is_video_playing
    if not is_video_playing:
        is_video_playing = True
        video_thread = threading.Thread(target=process_video_feed)
        video_thread.start()

# Function to stop the video feed
def stop_video_feed():
    global is_video_playing
    is_video_playing = False

# Function to process the video feed
def process_video_feed():
    while is_video_playing:
        frame = get_frame_from_stream(STREAM_URL)
        if frame is not None:
            frame = cv2.resize(frame, (640, 640))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame = frame / 255.0
            # frame = (frame * 255).astype(np.uint8)
            # frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(device)
            frame_tensor = torch.from_numpy(frame / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            res = seg_model(frame_tensor)
            box_multi_list = []
            centr_pt_cur_fr = []

            result = res[0]
            classes = np.array(result.boxes.cls.cpu(), dtype="int")
            confidence = np.array(result.boxes.conf.cpu())
            bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")

            idx = []
            for i in range(0, len(classes)):
                if classes[i] == 0:
                    idx.append(i)

            bbox = []
            for i in idx:
                temp = bboxes[i]
                bbox.append(temp)

                box_multi_list = [arr.tolist() for arr in bbox]

            for box in box_multi_list:
                (x, y, x2, y2) = box

                cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
                cx = int((x+x2)/2)
                cy = int((y+y2)/2)
                centr_pt_cur_fr.append((cx, cy))
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            head_count = len(centr_pt_cur_fr)
            count_var = head_count

            text = f'Head Count: {head_count}'
            cv2.putText(frame, text, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            frame = cv2.resize(frame, (640, 480))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(image=frame)
            video_label.config(image=frame)
            video_label.image = frame
        else:
            print("Failed to retrieve frame from the stream.")
            break

options = ["Stream 1", "Stream 2", "Stream 3", "Stream 4"]

def dropdown_selected(event):
    global STREAM_URL
    selected_option = dropdown_var.get()
    global STREAM_URL
    if selected_option == "Stream 1":
        STREAM_URL = STREAM_URLS[0]
    elif selected_option == "Stream 2":
        STREAM_URL = STREAM_URLS[1]
    elif selected_option == "Stream 3":
        STREAM_URL = STREAM_URLS[2]
    elif selected_option == "Stream 4":
        STREAM_URL = STREAM_URLS[3]
    # Stop current video feed (if any)
    stop_video_feed()
    # Start video feed with the selected URL
    start_video_feed()

# Create and configure the GUI components
main_frame = ttk.Frame(root, padding=10)
main_frame.grid(column=0, row=0, sticky=(tk.W, tk.N, tk.E, tk.S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# Create a custom style for buttons
style = ttk.Style()
style.configure("Professional.TButton", foreground="white", background="#007acc", padding=(10, 5))

# Column 1: Dropdown
dropdown_var = tk.StringVar()
dropdown = ttk.Combobox(main_frame, textvariable=dropdown_var, values=options)
dropdown.set(options[0])
dropdown.grid(column=0, row=0, padx=5, pady=5, sticky=tk.W)
dropdown.bind("<<ComboboxSelected>>", dropdown_selected)

# Column 2: Buttons
button_start = ttk.Button(main_frame, text="Start Video", command=start_video_feed, style="Professional.TButton")
button_start.grid(column=1, row=0, padx=5, pady=5, sticky=tk.W)

button_stop = ttk.Button(main_frame, text="Stop Video", command=stop_video_feed, style="Professional.TButton")
button_stop.grid(column=1, row=1, padx=5, pady=5, sticky=tk.W)

button_extra1 = ttk.Button(main_frame, text="Extra Button 1", style="Professional.TButton")
button_extra1.grid(column=1, row=2, padx=5, pady=5, sticky=tk.W)

button_extra2 = ttk.Button(main_frame, text="Extra Button 2", style="Professional.TButton")
button_extra2.grid(column=1, row=3, padx=5, pady=5, sticky=tk.W)

# Column 3: Video Feed
video_label = ttk.Label(main_frame)
video_label.grid(column=2, row=0, rowspan=5, padx=10, pady=10, sticky=(tk.N, tk.W, tk.E, tk.S))

# Initialize video playing status
is_video_playing = False

# Start the Tkinter main loop
root.mainloop()
