from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dropout, Dense
from keras.models import Model, Sequential
import numpy as np
import imageio.core.util
from facenet_pytorch import MTCNN
from PIL import Image
import cv2
from keras.optimizers import Adam, Nadam
from keras.applications.xception import Xception
from random import shuffle, sample
from os import listdir
import glob
from os.path import join
import torch, gc
import efficientnet.keras as efn
from keras import initializers

def ignore_warnings(*args, **kwargs):
    pass


def cnn_model(img_size, weights):

    input_size = (img_size, img_size, 3)
    baseModel = efn.EfficientNetB7(
        weights="imagenet",
        include_top=False,
        input_shape=input_size,
        pooling='max'
    )

    model = Sequential()
    model.add(baseModel)
    model.add(Dense(units=512, activation='relu', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                    bias_initializer='zeros', name="fc1"))
    model.add(Dropout(0.5))
    model.add(Dense(units=128, activation='relu', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                    bias_initializer='zeros'))
    model.add(Dense(units=2, activation='softmax', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                    bias_initializer='zeros'))
    model.summary()

    for layer in baseModel.layers:
        layer.trainable = True

    model.load_weights(weights)

    optimizer = Adam(
        lr=0.0001
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    model_lstm = Model(
        inputs=model.input,
        outputs=model.get_layer("fc1").output
    )
    return model_lstm

def main():
    imageio.core.util._precision_warn = ignore_warnings

    # Number of frames to be taken into consideration
    seq_length=25
    load_weights_name="./models/EfficientNetB7_w_NT_299_batch_10_v2.hdf5"
    image_size=299 #batch size

    # MTCNN face extraction from frames

    # Create face detector
    mtcnn = MTCNN(
        image_size=image_size,
        margin=40,
        select_largest=False,
        post_process=False,
        device="cuda:0"
    )

    train_dir = "./1. NeuralTextures/"

    videos = []
    videos += glob.glob(join(train_dir, "1", "*.mp4"))
    videos += glob.glob(join(train_dir, "0", "*.mp4"))

    test_videos = sample(videos, 280)
    shuffle(test_videos)

    temp_list = []
    for vid in videos:
        if vid not in test_videos:
            temp_list.append(vid)
    del videos
    videos = temp_list
    del temp_list

    print("length of videos", len(videos))
    shuffle(videos)

    # Loading model for feature extraction
    model = cnn_model(
        img_size=image_size,
        weights=load_weights_name,
    )

    features = []
    labels = []
    counter = 0

    for video in videos:
        cap = cv2.VideoCapture(video)
        labels += [int(video.split("/")[-2])]

        batches = []

        while cap.isOpened() and len(batches) < seq_length:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            if h >= 1080 and w >= 1920:
                frame = cv2.resize(
                    frame,
                    (640, 480),
                    interpolation=cv2.INTER_AREA
                )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            face = mtcnn(frame)

            try:
                face = face.permute(1, 2, 0).int().numpy()
                batches.append(face)

                del frame, face
                gc.collect()
                torch.cuda.empty_cache()

            except AttributeError:
                print("Image Skipping")

        cap.release()

        batches = np.array(batches).astype("float32")
        batches /= 255

        # fc layer feature generation
        predictions = model.predict(batches)
        features += [predictions]
        print("Number of videos done:", counter)

        if counter % 500 == 0:
            print("Number of videos done:", counter)
            print(np.array(features).shape, np.array(labels).shape)
            np.save("./dataset_npy/lstm_NT_train_data_{}.npy".format(str(counter)), np.array(features))
            np.save("./dataset_npy/lstm_NT_train_label_{}.npy".format(str(counter)), np.array(labels))

            del features, labels
            features = []
            labels = []

        counter += 1

    labels = np.array(labels)
    features = np.array(features)
    print(features.shape, labels.shape)

    np.save("./dataset_npy/lstm_NT_train_data_{}.npy".format(str(counter)), features)
    np.save("./dataset_npy/lstm_NT_train_labels_{}.npy".format(str(counter)), labels)

    del features, labels, videos

# ------------------------------------------- Test Data -------------------------------------------------
    features = []
    labels = []
    counter = 0

    for video in test_videos:
        cap = cv2.VideoCapture(video)
        labels += [int(video.split("/")[-2])]

        batches = []

        while cap.isOpened() and len(batches) < seq_length:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            if h >= 1080 and w >= 1920:
                frame = cv2.resize(
                    frame,
                    (640, 480),
                    interpolation=cv2.INTER_AREA
                )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            face = mtcnn(frame)

            try:
                face = face.permute(1, 2, 0).int().numpy()
                batches.append(face)

                del frame, face
                gc.collect()
                torch.cuda.empty_cache()

            except AttributeError:
                print("Image Skipping")

        cap.release()

        batches = np.array(batches).astype("float32")
        batches /= 255

        # fc layer feature generation
        predictions = model.predict(batches)
        features += [predictions]
        print("Number of videos done:", counter)

        if counter % 200 == 0:
            print("Number of videos done:", counter)
            print(np.array(features).shape, np.array(labels).shape)

        counter += 1

    labels = np.array(labels)
    features = np.array(features)
    print(features.shape, labels.shape)

    np.save("./dataset_npy/lstm_NT_test_data.npy", features)
    np.save("./dataset_npy/lstm_NT_test_labels.npy", labels)

    del features, labels

if __name__ == "__main__":
    main()

