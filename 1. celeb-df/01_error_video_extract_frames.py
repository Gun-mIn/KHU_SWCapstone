# frame 캡쳐가 제대로 되지 않은 영상 해결
import cv2
from os import makedirs
from os.path import join, exists
import glob


video_path = "./1. celeb-df/0/id27_0007.mp4"

cap = cv2.VideoCapture(video_path)

frameId = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("can't open the video")
        break

    filename = (
            "./2. train_frames/0/video_277"
            + "/image_"
            + str(int(frameId) + 1)
            + ".jpg"
    )
    cv2.imwrite(filename, frame)
    
    frameId += 1

cap.release()
