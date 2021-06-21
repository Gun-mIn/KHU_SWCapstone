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
source_frames_folders = ["./train_frames/0", "./train_frames/1"]
# Destination location where faces cropped out from images will be saved
dest_faces = "./train_face/"


for i in source_frames_folders:
    counter = 0
    for j in listdir(i):
        if i.find("0") != -1:
            dest_faces_folder = "{}0".format(dest_faces)
        else:
            dest_faces_folder = "{}1".format(dest_faces)

        imgs = glob.glob(join(i, j, "*.jpg"))

        if counter % 1000 == 0:
            print("Number of videos done:", counter)
        if not exists(join(dest_faces_folder, j)):
            makedirs(join(dest_faces_folder, j))

        for k in imgs:
            frame = cv2.imread(k)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            faces = mtcnn(frame)
            flag=0
            print(k)
            try:
                for face in faces:
                    try:
                        flag += 1
                        imsave(
                            join(dest_faces_folder, j, k.split("/")[-1].replace(".jpg", "_" + str(flag)) + ".jpg"),
                            face.permute(1, 2, 0).int().numpy()
                        )

                    except AttributeError:
                        print("Image skipping")

                    except TypeError:
                        print("None Type object is iterable")
            except TypeError:
                print("None Type object is iterable")
            flag=0
    counter += 1

