# importing libraries
from flask import Flask, render_template, Response, request
import cv2
import os
import signal
from ultralytics import YOLO
import numpy as np
from pathlib import Path

# global variable stores the people count
count_var = 0

# creating flask app
app = Flask(__name__)

# Define the segmentation model name
SEG_MODEL_NAME = "yolov8n-seg"

# Create a directory for models
models_dir = Path('models')
models_dir.mkdir(exist_ok=True)

def generate_frames(ip_address):

    # Access the global variable
    global count_var

    # Load the segmentation model
    seg_model = YOLO(models_dir / f'{SEG_MODEL_NAME}.pt')

    # Open the video capture
    cap = cv2.VideoCapture(ip_address)

    while True:

        # ---------- capturing frames-----------#
        ret , frame = cap.read()
        if not ret :
            break

        # Resize frames
        frame = cv2.resize(frame, (1400, 800))

        # List that stores the centroids of the current frame
        centr_pt_cur_fr = []

        # Perform segmentation on the frame
        res = seg_model(frame)

        # Get class labels, confidence levels, and bounding boxes
        classes = np.array(res[0].boxes.cls.cpu(), dtype="int")
        confidence = np.array(res[0].boxes.conf.cpu())
        bboxes = np.array(res[0].boxes.xyxy.cpu(), dtype="int")

        # Get indexes of the detections containing persons
        idx = [i for i, c in enumerate(classes) if c == 0]

        # Bounding boxes for person detections
        bbox = [bboxes[i] for i in idx]

        # Convert bbox to a multidimensional list
        box_multi_list = [arr.tolist() for arr in bbox]

        # Draw bounding boxes
        for box in box_multi_list:
            (x, y, x2, y2) = box
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
            cx = int((x + x2) / 2)
            cy = int((y + y2) / 2)
            centr_pt_cur_fr.append((cx, cy))
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # Count total people in the footage
        head_count = len(centr_pt_cur_fr)
        count_var = head_count

        # Display the face count on the screen
        cv2.putText(frame, f'Head Count: {head_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        # If 'q' is pressed, break the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # Convert the frame to JPEG and yield it
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

        
# home route
@app.route("/")
def index():
    return render_template('index.html')

# video feed route
@app.route("/video_feed")
def video_feed():
    ip_address = request.args.get('ip')
    if ip_address == "0":
        ip_address = 0
    return Response(generate_frames(ip_address), mimetype='multipart/x-mixed-replace; boundary=frame')

# video stop feed route
@app.route("/stop_feed")
def stop_feed():
    os.kill(os.getpid(), signal.SIGINT)
    return "feed stopped!"

# face count route
@app.route("/count")
def count():
    return str(count_var)

# classroom route
@app.route("/classroom", methods = ['GET', 'POST'])
def classroom():

    # logic for input field validation
    if request.method == 'POST':
        
        if (request.form['ip'] == ''):
            inv_feed ="No Video-Feed!"
            return render_template('classroom.html',var2 = inv_feed)
        
        else:
            ip_address = request.form['ip']
            ip_vd_feed = "Video-Feed"
            return render_template('classroom.html', ip_address = ip_address, var2 = ip_vd_feed)
    
    if request.method == 'GET':
        return render_template('classroom.html')


# about page route
@app.route("/about")
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, port=8000)
