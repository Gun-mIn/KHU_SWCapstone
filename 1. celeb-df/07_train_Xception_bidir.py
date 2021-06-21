from keras.layers.core import Dense
from keras.layers import Input, LSTM, Bidirectional
from keras.models import Model
from keras.optimizers import SGD, Nadam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.layers import GRU
from keras import utils
import numpy as np
import time
from keras.engine import InputSpec
from keras.engine.topology import Layer
from matplotlib import pyplot as plt

def rnn_models(train_data):
    #weights = "./trained_wts/EfficientNetB7_w_Celeb-DF_160_batch_10_v2.hdf5"

    main_input = Input(
        shape=(train_data.shape[1],
               train_data.shape[2]),
        name="main_input"
    )

    headModel = Bidirectional(LSTM(256, return_sequences=True))(main_input)
    headModel = LSTM(32)(headModel)


    predictions = Dense(
        2,
        activation="softmax",
        kernel_initializer="he_uniform"
    )(headModel)
    model = Model(inputs=main_input, outputs=predictions)
    #model.load_weights(weights)

    # Model compilation
    optimizer = Nadam(
        lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    return model


def main():
    start = time.time()

    epochs=30
    version=1
    weights_save_name = "bidir_LSTM_CF_{}".format(str(version))
    batch_size=32

    # Training dataset loading
    train_data1 = np.load("./dataset_npy/lstm_CF_train_data_0.npy", allow_pickle=True)
    train_label1 = np.load("./dataset_npy/lstm_CF_train_label_0.npy", allow_pickle=True)

    train_data2 = np.load("./dataset_npy/lstm_CF_train_data_1000.npy", allow_pickle=True)
    train_label2 = np.load("./dataset_npy/lstm_CF_train_label_1000.npy", allow_pickle=True)

    train_data = np.concatenate((train_data1, train_data2), axis=0)
    train_label = np.concatenate((train_label1, train_label2), axis=0)

    del train_data1, train_data2, train_label1, train_label2

    train_data3 = np.load("./dataset_npy/lstm_CF_train_data_2000.npy", allow_pickle=True)
    train_label3 = np.load("./dataset_npy/lstm_CF_train_label_2000.npy", allow_pickle=True)

    train_data = np.concatenate((train_data, train_data3), axis=0)
    train_label = np.concatenate((train_label, train_label3), axis=0)

    del train_data3, train_label3

    train_data4 = np.load("./dataset_npy/lstm_CF_train_data_3000.npy", allow_pickle=True)
    train_label4 = np.load("./dataset_npy/lstm_CF_train_label_3000.npy", allow_pickle=True)

    train_data = np.concatenate((train_data, train_data4), axis=0)
    train_label = np.concatenate((train_label, train_label4), axis=0)

    del train_data4, train_label4

    train_data5 = np.load("./dataset_npy/lstm_CF_train_data_5000.npy", allow_pickle=True)
    train_label5 = np.load("./dataset_npy/lstm_CF_train_label_5000.npy", allow_pickle=True)

    train_data = np.concatenate((train_data, train_data5), axis=0)
    train_label = np.concatenate((train_label, train_label5), axis=0)

    del train_data5, train_label5

    train_data6 = np.load("./dataset_npy/lstm_CF_train_data_5711.npy", allow_pickle=True)
    train_label6 = np.load("./dataset_npy/lstm_CF_train_labels_5711.npy", allow_pickle=True)

    train_data = np.concatenate((train_data, train_data6), axis=0)
    train_label = np.concatenate((train_label, train_label6), axis=0)

    del train_data6, train_label6

    print(train_data.shape, train_label.shape)
    train_label = utils.to_categorical(train_label)

    test_data = np.load("./dataset_npy/lstm_CF_test_data.npy", allow_pickle=True)
    test_label = np.load("./dataset_npy/lstm_CF_test_labels.npy", allow_pickle=True)
    test_label = utils.to_categorical(test_label)
    print(test_data.shape, test_label.shape)

    print("Dataset Loaded...")

    # Train validation split
    trainX, valX, trainY, valY = train_test_split(
        train_data, train_label, shuffle=True, test_size=0.2
    )

    model = rnn_models(train_data)

    trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    )
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])
    )

    # Number of trainable and non-trainable parameters
    print("Total params: {:,}".format(trainable_count + non_trainable_count))
    print("Trainable params: {:,}".format(trainable_count))
    print("Non-trainable params: {:,}".format(non_trainable_count))

    # Keras backend
    model_checkpoint = ModelCheckpoint(
        "trained_wts/" + weights_save_name + ".hdf5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )

    stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=0)

    print("Training is going to start in 3... 2... 1... ")

    # Model training
    H = model.fit(
        trainX,
        trainY,
        validation_data=(valX, valY),
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        callbacks=[model_checkpoint, stopping],
    )

    # plot the training loss and accuracy
    results = model.evaluate(test_data, test_label)
    print("Test acc, loss : ", results)

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


if __name__ == '__main__':
    main()