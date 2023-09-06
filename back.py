# importing libraries
from flask import Flask, render_template, Response, request
import cv2
import os
import signal
from ultralytics import yolo
import numpy as np
from pathlib import Path
import requests
from PIL import Image


# global variable stores the people count
count_var = 0

# creating flask app
app = Flask(__name__)

# home route
@app.route("/")
def index():
    return render_template('index.html')


models_dir = Path('models')
models_dir.mkdir(exist_ok=True)

def get_frame_from_stream(url: str) -> np.ndarray:
    response = requests.get(url)
    if response.status_code == 200:
        frame = np.asarray(bytearray(response.content), dtype=np.uint8)
        return cv2.imdecode(frame, cv2.IMREAD_COLOR)
    else:
        return None

def generate_frames(ip_address):

    # access the global variable
    global count_var

    # Replace the IMAGE_PATH with the URL of the video stream
    STREAM_URL = ip_address
    # STREAM_URL = "http://192.168.137.156:8080/video?type=some.mjpeg"

    SEG_MODEL_NAME = "yolov8n-seg"
    seg_model = YOLO(models_dir / f'{SEG_MODEL_NAME}.pt')

    while True:

         # Get the frame from the stream
        frame = get_frame_from_stream(STREAM_URL)

        if frame is None:
            print("Failed to retrieve frame from the stream.")
            break

        # Perform inference on the frame
        res = seg_model(frame)

        result_pil_image = Image.fromarray(res[0].plot()[:, :, ::-1])

        # Convert the PIL image to an array to use with OpenCV
        result_cv_image = np.array(result_pil_image)

        # Display the result in the OpenCV window
        cv2.imshow("Result Image", result_cv_image)

        # Wait for 1 second (1000 milliseconds) before displaying the next frame
        cv2.waitKey(500)

        # if the q is pressed the the loop is broken
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    

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