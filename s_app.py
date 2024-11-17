import streamlit as st
import pandas as pd
import numpy as np
import cv2
import math
import cv2, queue, threading, time
from ultralytics import RTDETR

model_path = r'C:\Users\U615994\OneDrive - IBERDROLA S.A\General - NEO · Real Time Smart Safety Eye\Models\Phase 1\reported_models\RT-DETRx_best.pt'
model = RTDETR(model_path)

classNames  = ['Air_basket','Arm','Belt','Black_boot','Face','Glove','Goggles','Green_pants','Green_shirt','Hand',
               'Head','Helmet','Ladder','Non_black_boot','Non_green_pants','Non_green_shirt','Sleeves']

class_dict  = {'Air_basket': 0,
                'Arm': 1,
                'Belt': 2,
                'Black_boot': 3,
                'Face': 4,
                'Glove': 5,
                'Goggles': 6,
                'Green_pants': 7,
                'Green_shirt': 8,
                'Hand': 9, 
               'Head': 10,
               'Helmet': 11,
               'Ladder': 12,
               'Non_black_boot': 13,
               'Non_green_pants': 14,
               'Non_green_shirt': 15,
               'Sleeves': 16}


class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get(timeout=5)
  

def predict(img, classes):
    results = model(img, classes=classes,stream=True, )
	# 
    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
    return img

# from streamlit_webrtc import webrtc_streamer
st.set_page_config(layout="wide")
st.title('RTSSE - Real-Time Smart Saftey Eye')
# on = st.toggle("Show Violations")

# if on:
#     st.write("Violations")



col1, col2 = st.columns([2, 1])


ctrl_container =  col2.container(border=True)
ctrl_container.subheader("Control View")

classes = ['Air_basket','Arm','Belt','Black_boot','Face','Glove','Goggles',
    'Green_pants','Green_shirt','Hand','Head','Helmet','Ladder','Non_black_boot',
    'Non_green_pants','Non_green_shirt','Sleeves']

options = ctrl_container.multiselect(
    "Classes to detect",
    classes,
    # classes
    # ["Green", "Yellow", "Red", "Blue"],
    # ["Yellow", "Red"],
)

indices = [class_dict[option] for option in options]
# st.write(indices)

video_container = col1.container(border=True)
video_container.subheader("Real-time Video")

# cap = cv2.VideoCapture('rtsp://192.168.1.3/live')
cap = VideoCapture('rtsp://192.168.1.3/live')

frame_placeholder = video_container.empty()
stop_button_pressed = video_container.button("Stop")

while not stop_button_pressed: # cap.isOpened() and :
    frame = cap.read()
    # if not ret:
    #     video_container.write("Video Capture Ended")
    #     break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if len(indices) > 0:
        frame = predict(frame, indices)
    frame_placeholder.image(frame,channels="RGB")
    if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
        break
cap.release()
