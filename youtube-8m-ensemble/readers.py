# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides readers configured for different datasets."""

import sys
import tensorflow as tf
import utils

from tensorflow import logging

def resize_axis(tensor, axis, new_size, fill_value=0):
  tensor = tf.convert_to_tensor(tensor)
  shape = tf.unstack(tf.shape(tensor))

  pad_shape = shape[:]
  pad_shape[axis] = tf.maximum(0, new_size - shape[axis])

  shape[axis] = tf.minimum(shape[axis], new_size)
  shape = tf.stack(shape)

  resized = tf.concat([
      tf.slice(tensor, tf.zeros_like(shape), shape),
      tf.fill(tf.stack(pad_shape), tf.cast(fill_value, tensor.dtype))
  ], axis)

  # Update shape.
  new_shape = tensor.get_shape().as_list()  # A copy is being made.
  new_shape[axis] = new_size
  resized.set_shape(new_shape)
  return resized

class BaseReader(object):
  """Inherit from this class when implementing new readers."""

  def prepare_reader(self, unused_filename_queue):
    """Create a thread for generating prediction and label tensors."""
    raise NotImplementedError()


class EnsembleReader(BaseReader):

  def __init__(self,
               num_classes=4716,
               feature_sizes=[1024],
               feature_names=["mean_inc3"]):

    assert len(feature_names) == len(feature_sizes), \
        "length of feature_names (={}) != length of feature_sizes (={})".format( \
        len(feature_names), len(feature_sizes))

    self.num_classes = num_classes
    self.feature_sizes = feature_sizes
    self.feature_names = feature_names

  def prepare_reader(self, filename_queue, batch_size=1024):

    reader = tf.TFRecordReader()
    _, serialized_examples = reader.read_up_to(filename_queue, batch_size)

    # set the mapping from the fields to data types in the proto
    num_features = len(self.feature_names)
    assert num_features > 0, "self.feature_names is empty!"
    assert len(self.feature_names) == len(self.feature_sizes), \
    "length of feature_names (={}) != length of feature_sizes (={})".format( \
    len(self.feature_names), len(self.feature_sizes))

    feature_map = {"video_id": tf.FixedLenFeature([], tf.string),
                   "labels": tf.VarLenFeature(tf.int64)}
    for feature_index in range(num_features):
      feature_map[self.feature_names[feature_index]] = tf.FixedLenFeature(
          [self.feature_sizes[feature_index]], tf.float32)

    features = tf.parse_example(serialized_examples, features=feature_map)
    labels = tf.sparse_to_indicator(features["labels"], self.num_classes)
    labels.set_shape([None, self.num_classes])
    concatenated_features = tf.concat([
        features[feature_name] for feature_name in self.feature_names], 1)

    return features["video_id"], concatenated_features, labels, tf.ones([tf.shape(serialized_examples)[0]])

class EnsembleFrameReader(BaseReader):

  def __init__(self,
               num_classes=4716,
               feature_sizes=[1024],
               feature_names=["mean_inc3"],
               max_frames=300):

    assert len(feature_names) == len(feature_sizes), \
        "length of feature_names (={}) != length of feature_sizes (={})".format( \
        len(feature_names), len(feature_sizes))

    self.num_classes = num_classes
    self.feature_sizes = feature_sizes
    self.feature_names = feature_names
    self.max_frames = max_frames

  def get_video_matrix(self,
                       features,
                       feature_size,
                       max_frames):

    reshaped_features = tf.reshape(features, [-1])
    num_frames = tf.minimum(tf.shape(reshaped_features)[0], max_frames)
    feature_matrix = resize_axis(reshaped_features, 0, max_frames, fill_value="")
    return feature_matrix, num_frames

  def prepare_reader(self, filename_queue):

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    contexts, features = tf.parse_single_sequence_example(
        serialized_example,
        context_features={
            "video_id": tf.FixedLenFeature([], tf.string),
            "labels": tf.VarLenFeature(tf.int64)},
        sequence_features={
            "rgb": tf.FixedLenSequenceFeature([], dtype=tf.string),
            "audio": tf.FixedLenSequenceFeature([], dtype=tf.string),
        })

    # read ground truth labels
    labels = (tf.cast(
        tf.sparse_to_dense(contexts["labels"].values, (self.num_classes,), 1,
            validate_indices=False),
        tf.bool))

    rgbs, num_frames = self.get_video_matrix(features["rgb"], 1024, self.max_frames)
    audios, num_frames = self.get_video_matrix(features["audio"], 1024, self.max_frames)

    batch_video_ids = tf.expand_dims(contexts["video_id"], 0)
    batch_rgbs = tf.expand_dims(rgbs, 0)
    batch_audios = tf.expand_dims(audios, 0)
    batch_labels = tf.expand_dims(labels, 0)
    batch_frames = tf.expand_dims(num_frames, 0)

    return batch_video_ids, batch_rgbs, batch_audios, batch_labels, batch_frames

