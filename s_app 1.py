import streamlit as st
import pandas as pd
import numpy as np
import cv2
import math
import cv2, queue, threading, time
from ultralytics import RTDETR

def plot_bbox(img, bbox):
   pass


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
  
def get_violations():
   pass
def get_predictions(img, classes):
    results = model(img, classes=classes,stream=True, conf=slide_conf/100)
	# 
    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            # if tgl_cls_name:
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])
            
            color = class_dict[classNames[cls]][1]#(255, 0, 0)

            if not tgl_violations:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # confidence
            # if tgl_conf:

            # object details
            org = [x1, y1-10]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            thickness = 1
            text = f'{classNames[cls]} '
            if not tgl_cls_name:
                text = ''
            if tgl_conf:
               text += f'{confidence}'

            cv2.putText(img, text, org, font, fontScale, color, thickness)

    return img


if __name__ == "__main__":
    model_path = r'C:\Users\U615994\OneDrive - IBERDROLA S.A\General - NEO · Real Time Smart Safety Eye\Models\Phase 1\reported_models\RT-DETRx_best.pt'
    model = RTDETR(model_path)

    classNames  = ['Air_basket','Arm','Belt','Black_boot','Face','Glove','Goggles','Green_pants','Green_shirt','Hand',
                'Head','Helmet','Ladder','Non_black_boot','Non_green_pants','Non_green_shirt','Sleeves']

    class_dict  = {'Air_basket': [0, [153, 195, 232]],
                    'Arm': [1, [150, 140, 122]],
                    'Belt': [2, [191, 245, 54]],
                    'Black_boot': [3, [0, 0, 0]],
                    'Face': [4, [173, 113, 3]],
                    'Glove': [5, [165, 44, 209]],
                    'Goggles': [6, [255,255,255]],
                    'Green_pants': [7,[84,227,122]],
                    'Green_shirt': [8, [56, 194, 93]],
                    'Hand': [9, [180, 199, 179]], 
                'Head': [10, [209, 100, 200]],
                'Helmet': [11, [246, 255, 71]],
                'Ladder': [12, [129, 135, 4]],
                'Non_black_boot': [13, [255, 0, 0]],
                'Non_green_pants': [14, [245, 2, 255]],
                'Non_green_shirt': [15, [0, 0, 255]],
                'Sleeves': [16, [255, 132, 0]]}



    # from streamlit_webrtc import webrtc_streamer
    st.set_page_config(layout="wide")
    st.title('RTSSE - Real-Time Smart Saftey Eye')
    # on = st.toggle("Show Violations")

    # if on:
    #     st.write("Violations")



    col1, col2 = st.columns([3, 1])


    ctrl_container =  col2.container(border=True)
    ctrl_container.subheader("Control Predictions")

    tgl_violations = ctrl_container.toggle("Show Violations")
    tgl_conf = ctrl_container.toggle("Show Confidence")
    tgl_cls_name = ctrl_container.toggle("Show Class Name")

    slide_conf = ctrl_container.slider("Confidence", 0, 100, 50)
    ctrl_container.write(f'Confidence {slide_conf}')


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

    indices = [class_dict[option][0] for option in options]


    # st.write(indices)

    video_container = col1.container(border=True)
    video_container.subheader("Real-time Video")

    cap = cv2.VideoCapture(0)#'rtsp://192.168.1.3/live')
    # cap = VideoCapture('rtsp://192.168.1.3/live')

    frame_placeholder = video_container.empty()
    stop_button_pressed = video_container.button("Stop")

    while not stop_button_pressed: # cap.isOpened() and :
        ret, frame = cap.read()
        if not ret:
            video_container.write("Video Capture Ended")
            break
        frame = cv2.flip(frame,1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if len(indices) > 0:
            frame = get_predictions(frame, indices)
        frame_placeholder.image(frame,channels="RGB",use_column_width='always')
        if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
            break

    cap.release()
