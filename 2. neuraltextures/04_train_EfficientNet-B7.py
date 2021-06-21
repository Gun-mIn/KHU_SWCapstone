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


# pip install -U git+https://github.com/qubvel/efficientnet
# 이전에 학습된 weight 유무(bool)
def cnn_model(img_size, weights):
    # 이전에 학습된 weight 파일 이름
    weight_name = "EfficientNetB7_w_NT_299_batch_10_v3"
    
    """
    Model definition using Xception net architecture
    """
    input_size = (img_size, img_size, 3)
    baseModel = efn.EfficientNetB7(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size, img_size, 3),
        pooling='max'
    )

    model = Sequential()
    model.add(baseModel)
    model.add(Dense(units=512, activation='relu', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer='zeros'))
    model.add(Dropout(0.5))
    model.add(Dense(units=128, activation='relu', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer='zeros'))
    model.add(Dense(units=2, activation='softmax', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer='zeros'))
    model.summary()

    if weights:
        model.load_weights("models/"+weight_name+".hdf5")

    # weight 동결 해제하고 trainable로 변경
    for layer in baseModel.layers:
        layer.trainable = True

    optimizer = Adam(
        lr=0.0001
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )
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
        except RuntimeError as e: # Memory growth must be set before GPUs have been initialized
            print(e)

    epochs=30
    image_size=299
    batch_size=10
    weights = True
    version = 4

    weights_save_name="EfficientNetB7_w_NT_" + str(image_size) + "_batch_" + str(batch_size) + "_v{}".format(str(version))

    # Training dataset loading
    train_data = np.load("./dataset_npy/train_data_NT_fpv_25_rev_299.npy")
    train_label = np.load("./dataset_npy/train_label_NT_fpv_25_rev_299.npy")

    print("Dataset Loaded...")

    # Train and validation split
    trainX, valX, trainY, valY = train_test_split(
        train_data, train_label, test_size=0.2, shuffle=True
    )
    print(trainX.shape, valX.shape, trainY.shape, valY.shape)

    # Train nad validation image data generator
    trainAug = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    valAug = ImageDataGenerator(rescale=1.0 / 255.0)
    
    model = cnn_model(img_size=image_size,weights=weights )

    # Number of trainable and non-trainable parameters
    trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    )
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])
    )

    print("Total params: {:,}".format(trainable_count + non_trainable_count))
    print("Trainable params: {:,}".format(trainable_count))
    print("Non-trainable params: {:,}".format(non_trainable_count))

    if not exists("./models"):
        makedirs("./models")
    if not exists("./training_logs"):
        makedirs("./training_logs")
    if not exists("./plots"):
        makedirs("./plots")

    # Keras backend
    model_checkpoint = ModelCheckpoint(
        "models/" + weights_save_name + ".hdf5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )

    stopping = EarlyStopping(monitor="val_loss", patience=5, verbose=1, mode='min')
    csv_logger = CSVLogger(
        "training_logs/EfficientNetB7_NT_{}_batch_{}_v{}.log".format(str(image_size), str(batch_size),str(version)),
        separator=",",
        append=True,
    )

    print("Training is going to start in 3... 2... 1... ")

    # Model Training
    
    H = model.fit_generator(
        trainAug.flow(trainX, trainY, batch_size=batch_size),
        steps_per_epoch=len(trainX) // batch_size,
        validation_data=valAug.flow(valX, valY),
        validation_steps=len(valX) // batch_size,
        epochs=epochs,
        callbacks=[model_checkpoint, stopping, csv_logger],
    )
    '''
    H = model.fit(
        trainX, trainY, batch_size=batch_size,
        #steps_per_epoch=len(trainX) // batch_size,
        #validation_data=valAug.flow(valX, valY),
        validation_data=(valX, valY),
        #validation_steps=len(valX) // batch_size,
        epochs=epochs,
        callbacks=[model_checkpoint, stopping, csv_logger],
    )
    '''

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = stopping.stopped_epoch + 1
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plots/EfficientNetB7_training_plot_NT_{}_batch_{}_v{}.png".format(str(image_size), str(batch_size), str(version)))
    
    
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
