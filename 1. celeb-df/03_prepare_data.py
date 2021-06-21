import cv2
import numpy as np
from keras.utils import np_utils
import glob
from os.path import join
from os import listdir
from random import shuffle, sample

img_size = 160
frames_per_video=25

# 0 : real
# 1 : fake
count_fake = 0
count_real = 0

train_path = ["./train_faces/0", "./train_faces/aug", "./3. train_faces/1"]

list_0 = [join(train_path[0], x) for x in listdir(train_path[0])]
list_1 = [join(train_path[1], x) for x in listdir(train_path[1])]
list_2 = [join(train_path[2], x) for x in listdir(train_path[2])]

list_0 = sample(list_0, 590)
list_1 = sample(list_1, 5049)
list_2 = sample(list_2, 5639)

video_list = list_0 + list_1 + list_2
shuffle(video_list)

c = 0

while True:
    if c == 2:
        break

    elif c == 0:
        vid_list = video_list[:5639]
    else:
        vid_list = video_list[5639:]

    shuffle(vid_list)

    train_data = []
    train_label = []

    count = 0

    images = []
    labels = []

    counter = 0

    for x in vid_list:
        img = glob.glob(join(x, "*.jpg"))
        img.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

        images += img[:frames_per_video]
        #print(images)

        label = [k.split("/")[2] for k in img]
        
        if label[0] == "aug":
            label = [0 for lab in label]
            
            
        labels += label[:frames_per_video]
        #print(labels)

        if counter % 1000 == 0:
            print("Number of files done:", counter)
        counter += 1


    for j, k in zip(images, labels):
        if(k == "0"):
            count_real += 1
        else:
            count_fake += 1

        img = cv2.imread(j)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(
            img, (img_size, img_size), interpolation=cv2.INTER_AREA
        )
        train_data.append(img)
        train_label += [k]

        if count % 10000 == 0:
            print("Number of files done:", count)
            print(count_fake, count_real)
        count += 1

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    train_label = np_utils.to_categorical(train_label)
    print(c, " -> train data : ", train_data.shape, "train label : " ,train_label.shape)


    np.save("./dataset_npy/train_data_CF_fpv_" + str(frames_per_video) + "_rev_" + str(img_size) + "_{}".format(str(c+1)), train_data)
    np.save("./dataset_npy/train_label_CF_fpv_" + str(frames_per_video) + "_rev_" + str(img_size) + "_{}".format(str(c+1)), train_label)

    del train_data, train_label, vid_list, images, labels
    
    c += 1


