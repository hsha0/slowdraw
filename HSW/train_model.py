from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

def get_num_classes():
  classes = []
  with tf.gfile.GFile(FLAGS.classes_file, "r") as f:
    classes = [x for x in f]
  num_classes = len(classes)
  return num_classes

def get_input_fn(mode, tfrecord_pattern, batch_size):
    print()

def create_estimator_and_specs(run_config):
    model_params = tf.contrib.training.HParams(
        batch_size=FLAGS.batch_size,
        num_classes=get_num_classes(),
        learning_rate=FLAGS.learning_rate,
        )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=model_params)

def main(unused_args):
    create_estimator_and_specs(
        run_config=tf.estimator.RunConfig(
            model_dir=FLAGS.model_dir,
            save_checkpoints_secs=300,
            save_summary_steps=100))
    print("Hi")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument(
        "--training_data",
        type=str,
        default="",
        help="Path to training data (tf.Example in TFRecord format)")

    parser.add_argument(
        "--eval_data",
        type=str,
        default="",
        help="Path to evaluation data (tf.Example in TFRecord format)")

    parser.add_argument(
        "--classes_file",
        type=str,
        default="",
        help="Path to a file with the classes - one class per line")

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate used for training.")

    parser.add_argument(
        "--steps",
        type=int,
        default=100000,
        help="Number of training steps.")

    parser.add_argument(
        "--model_dir",
        type=str,
        default="model_dir",
        help="Path for storing the model checkpoints.")

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size to use for training/evaluation.")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)