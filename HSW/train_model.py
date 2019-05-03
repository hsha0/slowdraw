import tensorflow as tf

def get_single_record(example_proto, mode):
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

def get_input_fn(mode, tfrecord_pattern, batch_size):
    
  def _get_input():
    dataset = tf.data.TFRecordDataset.list_files(tfrecord_pattern)
    # if mode == tf.estimator.ModeKeys.TRAIN:
    #   dataset = dataset.shuffle(buffer_size=10)
    # dataset = dataset.repeat()
    # Preprocesses 10 files concurrently and interleaves records from each file.
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=10,
        block_length=1)
    dataset = dataset.map(
        functools.partial(get_single_record, mode=mode),
        num_parallel_calls=10)
    dataset = dataset.prefetch(10000)
    if mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.shuffle(buffer_size=1000000)
    # Our inputs are variable length, so pad them.
    dataset = dataset.padded_batch(batch_size, padded_shapes=dataset.output_shapes)
    features, labels = dataset.make_one_shot_iterator().get_next()
    return features, labels

  return _get_input
        

def main(unused_args):
    pass

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--TFRecord",
      type=str,
      default="../dataset/training.tfrecord-00000-of-00010",
      help="Path to training data (tf.Example in TFRecord format)")
  parser.add_argument(
      "--eval_data",
      type=str,
      default="../dataset/eval.tfrecord-00000-of-00010",
      help="Path to evaluation data (tf.Example in TFRecord format)")
  parser.add_argument(
      "--classes_file",
      type=str,
      default="../dataset/training.tfrecord.classes",
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
      default="./",
      help="Path for storing the model checkpoints.")


  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)