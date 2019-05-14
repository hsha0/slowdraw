# Slow, draw!
Han Shao, Synthia Wang, Zeo Huang

## Strokes Dataset in TFRecord Format
Download the data in `TFRecord` from
[here](http://download.tensorflow.org/data/quickdraw_tutorial_dataset_v1.tar.gz) and upzip it.

## Images Dataset in Numpy Format
The images dataset in npy format can be download through command:

`gsutil cp gs://quickdraw_dataset/full/numpy_bitmap/* DESTINATION_DIRECTORY`

Then npy11000.py can be used to extract the first 11,000 drawings from 
npy file of each category. (To reduce the size of dataset)
## Recurrent Neural Networks

To run train_model.py, please use the command

`python3 train_model.py python3 train_model.py`
 
`--training_data=PATH_TO_DATA_DIRECTORY/training.tfrecord-00???-of-00010`

`--eval_data=PATH_TO_DATA_DIRECTORY/eval.tfrecord-00???-of-00010`

`--classes_file=PATH_TO_DATA_DIRECTORY/training.tfrecord.classes`
 
`--model_dir=PATH_TO_MODEL_DIRECTORY`
 
`--cell_type=cudnn_lstm` (Speed up training, optional)
 
`--batch_norm=True` (Speed up training, optional)

## 3-layer Convolutional Neural Network
To run model.py, please use the command
`python3 model.py --learning_rate=LEARNING_RATE --model_dir=
PATH_TO_MODEL_DIRECTORY --steps=STEPS`

The npy files should be put in the directory named npy11000.
## Custom AlexNet
To run model_v2.py, please use the command
`python3 model_v2.py --learning_rate=LEARNING_RATE 
--model_dir=PATH_TO_MODEL_DIRECTORY --steps=STEPS`

The npy files should be put in the directory named npy11000.


