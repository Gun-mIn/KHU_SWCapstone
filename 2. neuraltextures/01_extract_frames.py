import cv2
from os import makedirs
from os.path import join, exists
import glob

training_videos_folder = ["./NeuralTextures/0", "./NeuralTextures/1"]


for folder in training_videos_folder:
    videos_path = glob.glob(join(folder, "*.mp4"))
    folder_flag = folder.split("/")[2]
    counter = 0

    for video_path in videos_path:
        cap = cv2.VideoCapture(video_path)
        vid = video_path.split("/")[-1]
        vid = vid.split(".")[0]
        frameRate = cap.get(5)  # frame rate

        if not exists("./train_frames/" + str(folder_flag) + "/video_" + str(int(counter))):
            makedirs("./train_frames/" + str(folder_flag) + "/video_" + str(int(counter)))

        #
        while cap.isOpened():
            frameId = cap.get(1)  # current frame number
            ret, frame = cap.read()

            if not ret:
                print("can't open the video")
                break

            filename = (
                    "./train_frames/"
                    + folder_flag
                    + "/video_"
                    + str(int(counter))
                    + "/image_"
                    + str(int(frameId) + 1)
                    + ".jpg"
            )
            cv2.imwrite(filename, frame)

        cap.release()

        if counter % 100 == 0:
            print("Number of videos done:", counter)
        counter += 1