from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.core import Dropout, Dense
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split
import efficientnet.keras as efn
from matplotlib import pyplot as plt
from keras import backend as K
import numpy as np
import time
from os.path import exists
from os import makedirs
from keras import initializers
import tensorflow as tf

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
                    bias_initializer='zeros'))
    model.add(Dropout(0.5))
    model.add(Dense(units=128, activation='relu', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                    bias_initializer='zeros'))
    model.add(Dense(units=2, activation='softmax', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                    bias_initializer='zeros'))
    model.summary()

    optimizer = Adam(
        lr=0.0001
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    model.load_weights(weights)

    return model


def main():
    start = time.time()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:  # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:  # Memory growth must be set before GPUs have been initialized
            print(e)

    image_size = 299

    test_data = np.load("./dataset_npy/test_data_NT_fpv_25_rev_299_2.npy")
    test_label = np.load("./dataset_npy/test_label_NT_fpv_25_rev_299_2.npy")
    test_data = test_data.astype('float32') / 255
    test_label = test_label.astype('float32') / 255
    print("Dataset Loaded...")

    weights1 = "./models/EfficientNetB7_w_NT_299_batch_10_v4.hdf5"
    model = cnn_model(img_size=image_size, weights=weights1)
    results_v1 = model.evaluate(test_data, test_label)
    print("model v1  : ", results_v1)

    del model


    end = time.time()
    dur = end - start

    if dur < 60:
        print("Execution Time:", dur, "seconds")
    elif dur > 60 and dur < 3600:
        dur = dur / 60
        print("Execution Time:", dur, "minutes")
    else:
        dur = dur / (60 * 60)
        print("Execution Time:", dur, "hours")


if __name__ == "__main__":
    main()
