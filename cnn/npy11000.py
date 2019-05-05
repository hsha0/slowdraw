from os import listdir
import numpy as np

for file_name in listdir("npy"):
    data_one_class = np.load("npy/" + file_name)[:11000]
    np.save("npy11000/" + file_name, data_one_class)
