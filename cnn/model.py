from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
import numpy as np
from os import listdir


images = None
first = True
labels = None

int2labels = []
labels2int = {}
for file_name in listdir("npy"):
    data_one_class = np.load("npy/"+file_name)[:10000]
    if first:
        images = data_one_class
    else:
        images = np.concatenate((images, data_one_class))

    label = file_name.split("_")[3][:-4]
    int2labels.append(label)
    labels2int[label] = len(int2labels)-1

    label_one_class = np.full((data_one_class.shape[0],1), labels2int[label])
    if first:
        labels = label_one_class
        first = False
    else:
        labels = np.concatenate((labels, label_one_class))


#shuffle


model = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(images, labels, epochs=5)



