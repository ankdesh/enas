import os
import sys
import cPickle as pickle
import numpy as np
import tensorflow as tf


def _read_data(data_path, train_files):
  """Reads CIFAR-10 format data. Always returns NHWC format.

  Returns:
    images: np tensor of size [N, H, W, C]
    labels: np tensor of size [N]
  """
  images, labels = [], []
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  train_data = np.concatenate([train_data, train_data, train_data], axis = 1)
  eval_data = np.concatenate([eval_data, eval_data, eval_data], axis = 1)

  #for file_name in train_files:
  #  print file_name
  #  full_name = os.path.join(data_path, file_name)
  #  with open(full_name) as finp:
  #    data = pickle.load(finp)
  #    batch_images = data["data"].astype(np.float32) / 255.0
  #    batch_labels = np.array(data["labels"], dtype=np.int32)
  #    images.append(batch_images)
  #    labels.append(batch_labels)
  #images = np.concatenate(images, axis=0)
  #labels = np.concatenate(labels, axis=0)
  #images = np.reshape(images, [-1, 3, 32, 32])
  #images = np.transpose(images, [0, 2, 3, 1])

  #return images, labels

  train_data = np.reshape(train_data, [-1, 3, 28, 28])
  train_data = np.transpose(train_data, [0, 2, 3, 1])
  return train_data, train_labels  

def read_data(data_path, num_valids=5000):
  print "-" * 80
  print "Reading data"

  images, labels = {}, {}

  train_files = [
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
  ]
  test_file = [
    "test_batch",
  ]
  images["train"], labels["train"] = _read_data(data_path, train_files)

  if num_valids:
    images["valid"] = images["train"][-num_valids:]
    labels["valid"] = labels["train"][-num_valids:]

    images["train"] = images["train"][:-num_valids]
    labels["train"] = labels["train"][:-num_valids]
  else:
    images["valid"], labels["valid"] = None, None

  images["test"], labels["test"] = _read_data(data_path, test_file)

  print "Prepropcess: [subtract mean], [divide std]"
  mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
  std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)

  print "mean: {}".format(np.reshape(mean * 255.0, [-1]))
  print "std: {}".format(np.reshape(std * 255.0, [-1]))

  images["train"] = (images["train"] - mean) / std
  if num_valids:
    images["valid"] = (images["valid"] - mean) / std
  images["test"] = (images["test"] - mean) / std

  return images, labels

