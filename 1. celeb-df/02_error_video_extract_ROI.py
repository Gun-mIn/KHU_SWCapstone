# frame 캡쳐가 제대로 되지 않은 영상 해결
from facenet_pytorch import MTCNN
import cv2
from PIL import Image

from os import listdir, makedirs
import glob
from os.path import join, exists
from skimage.io import imsave


mtcnn = MTCNN(
    keep_all=True,
    margin=40,
    select_largest=False,
    post_process=False,
    device="cuda:0"
)

# Directory containing images respective to each video
source_frames_folders = "./2. train_frames/0/video_277"
# Destination location where faces cropped out from images will be saved
dest_faces = "./3. train_faces/0/video_277"

for j in listdir(source_frames_folders):
    imgs = source_frames_folders + "/" + j
    print(imgs)

    frame = cv2.imread(imgs)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    faces = mtcnn(frame)
    flag = 0
    print(imgs)
    try:
        for face in faces:
            try:
                flag += 1
                imsave(
                    join(dest_faces, j.replace(".jpg", "_" + str(flag)) + ".jpg"),
                    face.permute(1, 2, 0).int().numpy()
                )

            except AttributeError:
                print("Image skipping")

            except TypeError:
                print("None Type object is iterable")
    except TypeError:
        print("None Type object is iterable")

