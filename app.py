import streamlit as st
import cv2 as cv
import copy
from ultralytics import YOLO
import numpy as np



def calculate_angle(p1, p2, p3):
    # p1, p2, p3 are the points in format [x, y]
    # Calculate the vectors
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    
    # Calculate the angle in radians
    angle_rad = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    # Convert to degrees
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def camera(): 
    video_placeholder = st.empty()
    angle_placeholder = st.empty()
    status_placeholder = st.empty()
    model = YOLO('yolov8n-pose.pt')
    # Camera capture
    cap = cv.VideoCapture(-1)

    # Create a placeholder for the video
    

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv.flip(frame, 1)  # Mirror display


        # frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)


        results = model(frame, show=False, conf=0.3)[0]
        if results.keypoints.xy is not None and results.keypoints.conf is not None:



            conf_score = results.keypoints.conf[0]
            keypoints = results.keypoints.xy[0]


            if conf_score[3] > conf_score[4]:
                angle = calculate_angle(keypoints[3], keypoints[5], keypoints[11])
            else:
                angle = calculate_angle(keypoints[4], keypoints[6], keypoints[12])

            angle_placeholder.write(angle)
            if 200 > angle > 160:
                status_placeholder.write("thang lung")
            else:
                status_placeholder.write("ngoi sai cmnr")



            for keypoint_indx, keypoint in enumerate(keypoints):
                cv.putText(frame, str(keypoint_indx), (int(keypoint[0]), int(keypoint[1])),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        video_placeholder.image(frame, channels="BGR")


if __name__ == '__main__':



 

    camera()