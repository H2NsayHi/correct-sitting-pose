from ultralytics import YOLO
import cv2
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



image_path = '/home/h2n/poseyolo/portrait-of-adult-man-sit-pose-on-white-background-D33JNE.jpg'
# img = cv2.imread(image_path)


model = YOLO('yolov8n-pose.pt')


# results = model(source=0, show=True, conf=0.3, save=False)[0]


cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # If frame is not read successfully, break the loop
    if not ret:
        print("Error: Could not read frame from video capture")
        break

    results = model(frame, show=True, conf=0.3)[0]
    if results.keypoints.xy is not None and results.keypoints.conf is not None:



        conf_score = results.keypoints.conf[0]
        keypoints = results.keypoints.xy[0]


        if conf_score[3] > conf_score[4]:
            angle = calculate_angle(keypoints[3], keypoints[5], keypoints[11])
        else:
            angle = calculate_angle(keypoints[4], keypoints[6], keypoints[12])
        # print(calculate_angle(keypoints[3], keypoints[5], keypoints[11]))

        print(angle)
        if 200 > angle > 160:
            print("thang lung")
        else:
            print("ngoi sai cmnr")
    # cv2.imshow("Pose Analysis", frame)

    # # Exit on 'q' key press
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break
# print(detects)

cap.release()
cv2.destroyAllWindows()






# for keypoint_indx, keypoint in enumerate(keypoints):
#     cv2.putText(img, str(keypoint_indx), (int(keypoint[0]), int(keypoint[1])),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# cv2.imshow('img', img)
# cv2.waitKey(0)


