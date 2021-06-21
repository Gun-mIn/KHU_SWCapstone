# opencv를 이용한 data augmentation
import cv2
import numpy as np
from keras.utils import np_utils
import glob
from os.path import join, exists
from os import listdir, makedirs
from random import shuffle
import random


# 죄우 반전
def horizontal_flip(img, flag):
    if flag:
        return cv2.flip(img, 1)
    else:
        return img


# 밝기 조절
def brightness(img, gamma=2.5):
    """
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    """
    #img = cv2.add(img, (100, 100, 100, 0))

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    
    img = cv2.LUT(img, table)
    return img


# Average Blurring
def average(img):
    img = cv2.blur(img, (5,5))
    return img

# Gaussian Blurring
def gaussian(img):
    img = cv2.GaussianBlur(img, (5,5), 0)
    return img

# Median Blurring
def median(img):
    img = cv2.medianBlur(img, 5)
    return img

# Bilateral Filtering
def bilateral(img):
    img = cv2.bilateralFilter(img, 9, 75, 75)
    return img



img_size = 299
frames_per_video=25

# 0 : real
count_real = 0

train_path = ["./3. train_faces/0/"]

vid_list = [join(train_path[0], x) for x in listdir(train_path[0])]

images = []
img_tmp = []
img_list = []


for x in vid_list:
    print("vid : ", x)  # <-  ./3. train_faces/0/video_296_vFlip
    folder = x.split("/")[3]

    img = glob.glob(join(x, "*.jpg"))
    img.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

    print("img : \n", img)

    if len(folder) <= 10:
        for i in range(4):
            img_tmp += img[:frames_per_video]
            del img[:frames_per_video]

            print("img_tmp", img_tmp) # <- './3. train_faces/0/video_29/image_76_1.jpg'

            for tmp in img_tmp:
                video_name = tmp.split("/")[3] + "_{}".format(i)
                img_name = tmp.split("/")[4]
                tmp = "./train_faces/0/" + video_name + "/" + img_name
                print("tmp",tmp)
                img_list.append(tmp)

            images += img_list
            print("len(img_tmp) 2 - ", len(img_tmp))
            img_tmp = []
            img_list = []


print("Number of lists done --> {}".format(str(len(images))))

for img in images:
    print("-img-  ", img) # <-  ./3. train_faces/0/video_209_hFlip/image_94_1.jpg
    video_folder = img.split("/")[3]
    img_name = img.split("/")[4]

    video_origin = video_folder.split("_")[0] + "_" + video_folder.split("_")[1]

    origin = "./3. train_faces/0/" + str(video_origin) + "/" + img_name
    dest_folder = "./train_faces/0/" + str(video_folder) + "/"

    if not exists(dest_folder):
        makedirs(dest_folder)

    image = cv2.imread(origin)
    image = cv2.resize(
        image, (img_size, img_size), interpolation=cv2.INTER_AREA
    )
    print(dest_folder + img_name)
    cv2.imwrite(dest_folder + "/" + img_name, image)


del train_path, vid_list, images, img_tmp, img_list


#-----------------------------------------------------------


train_path = ["./train_faces/0/"]

vid_list = [join(train_path[0], x) for x in listdir(train_path[0])]

images = []
img_tmp = []

for x in vid_list:
    print("vid : ", x)
    folder = x.split("/")[3]

    img = glob.glob(join(x, "*.jpg"))
    img.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

    print("img : \n", img)
    img_tmp = random.sample(img, frames_per_video)
    images.append(img_tmp)


print("Number of lists done --> {}".format(str(len(images))))

for j in images:
    for num in range(6):
        if num == 0: # 좌우반전
            for img in j:
                video_folder = img.split("/")[3] + "_hFlip"
                img_name = img.split("/")[4]
                dest_folder = "./train_faces/aug/" + str(video_folder)

                if not exists(dest_folder):
                    makedirs(dest_folder)

                image = cv2.imread(img)
                image = horizontal_flip(image, True)
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(
                    image, (img_size, img_size), interpolation=cv2.INTER_AREA
                )
                #print(dest_folder + img_name)
                cv2.imwrite(dest_folder + "/" + img_name, image)
                
        elif num == 1: # 밝기 조절
            for img in j:
                video_folder = img.split("/")[3] + "_bright"
                img_name = img.split("/")[4]
                dest_folder = "./train_faces/aug/" + str(video_folder)

                if not exists(dest_folder):
                    makedirs(dest_folder)

                image = cv2.imread(img)
                image = brightness(image, 2.5)
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(
                    image, (img_size, img_size), interpolation=cv2.INTER_AREA
                )
                print(dest_folder + img_name)
                cv2.imwrite(dest_folder + "/" + img_name, image)
                
        elif num == 2: # Average Blurring
            for img in j:
                video_folder = img.split("/")[3] + "_average"
                img_name = img.split("/")[4]
                dest_folder = "./train_faces/aug/" + str(video_folder)

                if not exists(dest_folder):
                    makedirs(dest_folder)

                image = cv2.imread(img)
                image = average(image)
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(
                    image, (img_size, img_size), interpolation=cv2.INTER_AREA
                )
                print(dest_folder + img_name)
                cv2.imwrite(dest_folder + "/" + img_name, image)
                
        elif num == 3: # Gaussian Blurring
            for img in j:
                video_folder = img.split("/")[3] + "_gaussian"
                img_name = img.split("/")[4]
                dest_folder = "./train_faces/aug/" + str(video_folder)

                if not exists(dest_folder):
                    makedirs(dest_folder)

                image = cv2.imread(img)
                image = gaussian(image)
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(
                    image, (img_size, img_size), interpolation=cv2.INTER_AREA
                )
                print(dest_folder + img_name)
                cv2.imwrite(dest_folder + "/" + img_name, image)
                
        elif num == 4: # Median Blurring
            for img in j:
                video_folder = img.split("/")[3] + "_median"
                img_name = img.split("/")[4]
                dest_folder = "./train_faces/aug/" + str(video_folder)

                if not exists(dest_folder):
                    makedirs(dest_folder)

                image = cv2.imread(img)
                image = median(image)
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(
                    image, (img_size, img_size), interpolation=cv2.INTER_AREA
                )
                print(dest_folder + img_name)
                cv2.imwrite(dest_folder + "/" + img_name, image)
                
        else: # Bilateral Filtering
            for img in j:
                video_folder = img.split("/")[3] + "_bilateral"
                img_name = img.split("/")[4]
                dest_folder = "./train_faces/aug/" + str(video_folder)

                if not exists(dest_folder):
                    makedirs(dest_folder)

                image = cv2.imread(img)
                image = bilateral(image)
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(
                    image, (img_size, img_size), interpolation=cv2.INTER_AREA
                )
                print(dest_folder + img_name)
                cv2.imwrite(dest_folder + "/" + img_name, image)
