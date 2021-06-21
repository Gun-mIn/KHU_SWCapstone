import numpy as np
from sklearn.model_selection import train_test_split
from random import shuffle
from os import listdir
import glob
from os.path import join

train_data = np.load("./dataset_npy/train_data_CF_fpv_25_rev_160.npy")
train_label = np.load("./dataset_npy/train_label_CF_fpv_25_rev_160.npy")

trainX, testX, trainY, testY = train_test_split(
        train_data, train_label, test_size=0.08316, shuffle=True
    )

print("split1")
np.save("./dataset_npy/only_train_data_CF_160.npy", trainX)
np.save("./dataset_npy/only_train_label_CF_160.npy", trainY)
print(trainX.shape, trainY.shape)

np.save("./dataset_npy/only_test_data_CF_160.npy", testX)
np.save("./dataset_npy/only_test_label_CF_160.npy", testY)

print(testX.shape, testY.shape)
