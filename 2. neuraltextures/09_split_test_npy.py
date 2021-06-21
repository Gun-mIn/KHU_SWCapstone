import numpy as np
from sklearn.model_selection import train_test_split
from random import shuffle
from os import listdir
import glob
from os.path import join

train_data = np.load("./npy/train_data_NT_fpv_25_rev_299.npy")
train_label = np.load("./npy/train_label_NT_fpv_25_rev_299.npy")

trainX, testX, trainY, testY = train_test_split(
        train_data, train_label, test_size=0.14, shuffle=True
    )


np.save("./dataset_npy/train_data_NT_fpv_25_rev_299.npy", trainX)
np.save("./dataset_npy/train_label_NT_fpv_25_rev_299.npy", trainY)
np.save("./dataset_npy/test_data_NT_fpv_25_rev_299.npy", testX)
np.save("./dataset_npy/test_label_NT_fpv_25_rev_299.npy", testY)

print(trainX.shape, testX.shape, trainY.shape, testY.shape)
