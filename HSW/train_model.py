from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import ast

import tensorflow as tf


def get_num_classes():
    classes = []
    with tf.gfile.GFile(FLAGS.classes_file, "r") as f:
    classes = [x for x in f]
    num_classes = len(classes)
    return num_classes

def get_input_fn(mode, tfrecord_pattern, batch_size):
    """Creates an input_fn that stores all the data in memory.

    Args:
    mode: one of tf.contrib.learn.ModeKeys.{TRAIN, INFER, EVAL}
    tfrecord_pattern: path to a TF record file created using create_dataset.py.
    batch_size: the batch size to output.

    Returns:
    A valid input_fn for the model estimator.
    """

    def _parse_tfexample_fn(example_proto, mode):
        """Parse a single record which is expected to be a tensorflow.Example."""
        feature_to_type = {
            "ink": tf.VarLenFeature(dtype=tf.float32),
            "shape": tf.FixedLenFeature([2], dtype=tf.int64)
        }

        if mode != tf.estimator.ModeKeys.PREDICT:
        # The labels won't be available at inference time, so don't add them
        # to the list of feature_columns to be read.
            feature_to_type["class_index"] = tf.FixedLenFeature([1], dtype=tf.int64)

        parsed_features = tf.parse_single_example(example_proto, feature_to_type)
        labels = None
        if mode != tf.estimator.ModeKeys.PREDICT:
            labels = parsed_features["class_index"]
        parsed_features["ink"] = tf.sparse_tensor_to_dense(parsed_features["ink"])
        return parsed_features, labels

    def _input_fn():
        """Estimator `input_fn`.

        Returns:
         A tuple of:
         - Dictionary of string feature name to `Tensor`.
         - `Tensor` of target labels.
        """
        dataset = tf.data.TFRecordDataset.list_files(tfrecord_pattern)
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(buffer_size=10)
        dataset = dataset.repeat()
        # Preprocesses 10 files concurrently and interleaves records from each file.
        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=10,
            block_length=1)
        dataset = dataset.map(
            functools.partial(_parse_tfexample_fn, mode=mode),
            num_parallel_calls=10)
        dataset = dataset.prefetch(10000)
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(buffer_size=1000000)
        # Our inputs are variable length, so pad them.
        dataset = dataset.padded_batch(
            batch_size, padded_shapes=dataset.output_shapes)
        features, labels = dataset.make_one_shot_iterator().get_next()
        return features, labels

    return _input_fn


def model_fn(features, labels, mode, params):
    """Model function for RNN classifier.

    This function sets up a neural network which applies convolutional layers (as
    configured with params.num_conv and params.conv_len) to the input.
    The output of the convolutional layers is given to LSTM layers (as configured
    with params.num_layers and params.num_nodes).
    The final state of the all LSTM layers are concatenated and fed to a fully
    connected layer to obtain the final classification scores.

    Args:
        features: dictionary with keys: inks, lengths.
        labels: one hot encoded classes
        mode: one of tf.estimator.ModeKeys.{TRAIN, INFER, EVAL}
        params: a parameter dictionary with the following keys: num_layers,
        num_nodes, batch_size, num_conv, conv_len, num_classes, learning_rate.

    Returns:
        ModelFnOps for Estimator API.
    """

    def _get_input_tensors(features, labels):
        """Converts the input dict into inks, lengths, and labels tensors."""
        # features[ink] is a sparse tensor that is [8, batch_maxlen, 3]
        # inks will be a dense tensor of [8, maxlen, 3]
        # shapes is [batchsize, 2]
        shapes = features["shape"]
        # lengths will be [batch_size]
        lengths = tf.squeeze(
            tf.slice(shapes, begin=[0, 0], size=[params.batch_size, 1]))
        inks = tf.reshape(features["ink"], [params.batch_size, -1, 3])
        if labels is not None:
            labels = tf.squeeze(labels)
        return inks, lengths, labels

    # Build the model.
    inks, lengths, labels = _get_input_tensors(features, labels)

    convoluted = inks
    convoluted_input = convoluted

    #Normalize the batch
    convoluted_input = tf.layers.batch_normalization(
            convolved_input,
            training=(mode == tf.estimator.ModeKeys.TRAIN))
    convolved = tf.layers.conv1d(
      convolved_input,
      filters=64,
      kernel_size=5,
      activation=tf.nn.relu,
      strides=1,
      padding="same",
      name="conv1d_1d")

    #CUDNN cell is faster when running on GPU
    convolved = tf.transpose(convolved, [1, 0, 2])
    lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
      num_layers=params.num_layers,
      num_units=params.num_nodes,
      dropout=params.dropout if mode == tf.estimator.ModeKeys.TRAIN else 0.0,
      direction="bidirectional")
    outputs, _ = lstm(convolved)
    # Convert back from time-major outputs to batch-major outputs.
    outputs = tf.transpose(outputs, [1, 0, 2])

    #To create a compact, fixed-length embedding, we sum up the output of the LSTMs.
    # We first zero out the regions of the batch where the sequences have no data.
    mask = tf.tile(
      tf.expand_dims(tf.sequence_mask(lengths, tf.shape(outputs)[1]), 2),
      [1, 1, tf.shape(outputs)[2]])
    zero_outside = tf.where(mask, outputs, tf.zeros_like(outputs))
    outputs = tf.reduce_sum(zero_outside, axis=1)

    #Adds a fully connected layer.
    tf.layers.dense(outputs, params.num_classes)

    # Add the loss.
    cross_entropy = tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=logits))

    # Add the optimizer.
    train_op = tf.contrib.layers.optimize_loss(
      loss=cross_entropy,
      global_step=tf.train.get_global_step(),
      learning_rate=params.learning_rate,
      optimizer="Adam",
      # some gradient clipping stabilizes training in the beginning.
      clip_gradients=params.gradient_clipping_norm,
      summaries=["learning_rate", "loss", "gradients", "gradient_norm"])

    # Compute current predictions.
    predictions = tf.argmax(logits, axis=1)

    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions={"logits": logits, "predictions": predictions},
      loss=cross_entropy,
      train_op=train_op,
      eval_metric_ops={"accuracy": tf.metrics.accuracy(labels, predictions)})



def create_estimator_and_specs(run_config):
    """Creates an Experiment configuration based on the estimator and input fn."""
    model_params = tf.contrib.training.HParams(
      num_layers=FLAGS.num_layers,
      num_nodes=FLAGS.num_nodes,
      batch_size=FLAGS.batch_size,
      num_conv=ast.literal_eval(FLAGS.num_conv),
      conv_len=ast.literal_eval(FLAGS.conv_len),
      num_classes=get_num_classes(),
      learning_rate=FLAGS.learning_rate,
      gradient_clipping_norm=FLAGS.gradient_clipping_norm,
      cell_type=FLAGS.cell_type,
      batch_norm=FLAGS.batch_norm,
      dropout=FLAGS.dropout)

    estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config,
      params=model_params)

    train_spec = tf.estimator.TrainSpec(input_fn=get_input_fn(
      mode=tf.estimator.ModeKeys.TRAIN,
      tfrecord_pattern=FLAGS.training_data,
      batch_size=FLAGS.batch_size), max_steps=FLAGS.steps)

    eval_spec = tf.estimator.EvalSpec(input_fn=get_input_fn(
      mode=tf.estimator.ModeKeys.EVAL,
      tfrecord_pattern=FLAGS.eval_data,
      batch_size=FLAGS.batch_size))

    return estimator, train_spec, eval_spec


def main(unused_args):
    estimator, train_spec, eval_spec = create_estimator_and_specs(
      run_config=tf.estimator.RunConfig(
          model_dir=FLAGS.model_dir,
          save_checkpoints_secs=300,
          save_summary_steps=100))
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


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
      "--num_layers",
      type=int,
      default=3,
      help="Number of recurrent neural network layers.")
    parser.add_argument(
      "--num_nodes",
      type=int,
      default=128,
      help="Number of node per recurrent network layer.")
    parser.add_argument(
      "--num_conv",
      type=str,
      default="[48, 64, 96]",
      help="Number of conv layers along with number of filters per layer.")
    parser.add_argument(
      "--conv_len",
      type=str,
      default="[5, 5, 3]",
      help="Length of the convolution filters.")
    parser.add_argument(
      "--cell_type",
      type=str,
      default="lstm",
      help="Cell type used for rnn layers: cudnn_lstm, lstm or block_lstm.")
    parser.add_argument(
      "--batch_norm",
      type="bool",
      default="False",
      help="Whether to enable batch normalization or not.")
    parser.add_argument(
      "--learning_rate",
      type=float,
      default=0.0001,
      help="Learning rate used for training.")
    parser.add_argument(
      "--gradient_clipping_norm",
      type=float,
      default=9.0,
      help="Gradient clipping norm used during training.")
    parser.add_argument(
      "--dropout",
      type=float,
      default=0.3,
      help="Dropout used for convolutions and bidi lstm layers.")
    parser.add_argument(
      "--steps",
      type=int,
      default=100000,
      help="Number of training steps.")
    parser.add_argument(
      "--batch_size",
      type=int,
      default=8,
      help="Batch size to use for training/evaluation.")
    parser.add_argument(
      "--model_dir",
      type=str,
      default="",
      help="Path for storing the model checkpoints.")
    parser.add_argument(
      "--self_test",
      type="bool",
      default="False",
      help="Whether to enable batch normalization or not.")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
