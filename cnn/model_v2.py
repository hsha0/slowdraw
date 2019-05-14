'''
model_v2.py is the custom AlexNet for drawings classification.

The dataset it takes is represented by 28x28 bitmap in numpy format.

Author: Han Shao, Synthia Wang, Zoe Huang

Date: 05/14/2019

'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
import numpy as np
from os import listdir
import argparse
import sys


def main(unused_args):

    # initialization
    train_images = None
    eval_images = None
    first = True
    train_labels = None
    eval_labels = None

    int2labels = []
    labels2int = {}

    # read images from mpy files
    for file_name in listdir("npy11000"):

        # read 11000 drawings from one class
        data_one_class = np.load("npy11000/"+file_name)[:11000]

        # if this is the first file
        if first:
            # add 10000 images to train_images
            train_images = data_one_class[:10000]
            # add the rest 1000 images to eval_images
            eval_images = data_one_class[10000:]

        else:
            # concatenate 10000 images to train_images
            train_images = np.concatenate((train_images, data_one_class[:10000]))
            # concatenate 10000 images to eval_images
            eval_images = np.concatenate((eval_images, data_one_class[10000:]))

        # read label
        label = file_name[:-4]
        # add int to label map
        int2labels.append(label)
        # add label to int map
        labels2int[label] = len(int2labels)-1

        # create a list of labels with length 11000
        label_one_class = np.full((data_one_class.shape[0],1), labels2int[label])

        # if this is the first file
        if first:
            # add 10000 labels to train_labels
            train_labels = label_one_class[:10000]
            # add the rest 1000 labels to eval_labels
            eval_labels = label_one_class[10000:]
            # set first to false
            first = False

        else:
            # concatenate 10000 labels to train_labels
            train_labels = np.concatenate((train_labels, label_one_class[:10000]))
            # concatenate 1000 labels to eval_labels
            eval_labels = np.concatenate((eval_labels, label_one_class[10000:]))

    # shuffle the train images
    assert len(train_images) == len(train_labels)
    random_pmt = np.random.permutation(len(train_images))
    train_images=train_images[random_pmt]
    train_labels=train_labels[random_pmt]

    # function to create cnn model
    def cnn_model_fn(features, labels, mode):

        # Input Layer
        input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
        print(input_layer)
        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=256,
            kernel_size=[3, 3],
            padding="valid",
            activation=tf.nn.relu)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2, padding='valid')

        # Convolutional Layer #2 #3 #4
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=384,
            kernel_size=[3, 3],
            padding="valid",
            activation=tf.nn.relu)

        conv3 = tf.layers.conv2d(
            inputs=conv2,
            filters=384,
            kernel_size=[3, 3],
            padding="valid",
            activation=tf.nn.relu)

        conv4 = tf.layers.conv2d(
            inputs=conv3,
            filters=256,
            kernel_size=[3, 3],
            padding="valid",
            activation=tf.nn.relu)

        # Pooling Layer #2
        pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[3, 3], strides=2, padding='valid')
        pool2_flat = tf.reshape(pool2, [-1, 2 * 2 * 256])

        # Dense Layer #1
        dense = tf.layers.dense(inputs=pool2_flat, units=4096, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

        # Dense Layer #2
        dense2 = tf.layers.dense(inputs=dropout, units=4096, activation=tf.nn.relu)
        dropout2 = tf.layers.dropout(
            inputs=dense2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

        # Logits Layer
        logits = tf.layers.dense(inputs=dropout2, units=len(int2labels))

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])
        }

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    # divide train_images and eval_images by 255 to make them 0 to 1
    train_data = train_images/np.float32(255)
    eval_data = eval_images/np.float32(255)

    # create drawing classifier based on cnn model
    drawing_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=FLAGS.model_dir)

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # evaluate the model after every 1000 steps
    for i in range(0, FLAGS.steps//1000):
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=100,
            num_epochs=None,
            shuffle=True)

        drawing_classifier.train(
            input_fn=train_input_fn,
            steps=1000,
            hooks=[logging_hook])

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)

        eval_results = drawing_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)


if __name__ == "__main__":

    # command line arguments
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate used for training.")
    parser.add_argument(
        "--steps",
        type=int,
        default=200000,
        help="Number of training steps.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="Path for storing the model checkpoints.")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


