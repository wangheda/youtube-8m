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

"""Provides definitions for non-regularized training or test losses."""

import numpy as np
import tensorflow as tf
from tensorflow import flags

FLAGS = flags.FLAGS
flags.DEFINE_float("false_negative_punishment", 1.0, 
                   "punishment constant to 1 classified to 0")
flags.DEFINE_float("false_positive_punishment", 1.0, 
                   "punishment constant to 0 classified to 1")
flags.DEFINE_integer("num_classes", 4716,
                   "number of classes")

flags.DEFINE_float("support_loss_percent", 0.1,
                   "the part that support loss (in multi-task scenario) take in the whole loss function.")
flags.DEFINE_string("support_type", "vertical",
                   "type of support label, vertical or frequent or vertical,frequent.")
flags.DEFINE_integer("num_supports", 25, "Number of total support categories.")
flags.DEFINE_integer("num_verticals", 25, "Number of total vertical categories.")
flags.DEFINE_integer("num_frequents", 200, "Number of total frequent categories.")
flags.DEFINE_string("vertical_file", "resources/vertical.tsv", "Location of label-vertical mapping file.")

flags.DEFINE_float("batch_agreement", 0.1,
                   "the batch_agreement parameter")

flags.DEFINE_bool("label_smoothing", False,
                   "whether do label smoothing")
flags.DEFINE_float("label_smoothing_epsilon", 0.1,
                   "whether do label smoothing")

def smoothing(labels):
  print "label smoothing for", labels
  epsilon = FLAGS.label_smoothing_epsilon
  float_labels = tf.cast(labels, tf.float32)
  num_labels = tf.reduce_sum(float_labels, axis=1, keep_dims=True)
  K = float_labels.get_shape().as_list()[1]
  prior = num_labels / K
  smooth_labels = float_labels * (1.0 - epsilon) + prior * epsilon
  return smooth_labels

class BaseLoss(object):
  """Inherit from this class when implementing new losses."""

  def calculate_loss(self, unused_predictions, unused_labels, **unused_params):
    """Calculates the average loss of the examples in a mini-batch.

     Args:
      unused_predictions: a 2-d tensor storing the prediction scores, in which
        each row represents a sample in the mini-batch and each column
        represents a class.
      unused_labels: a 2-d tensor storing the labels, which has the same shape
        as the unused_predictions. The labels must be in the range of 0 and 1.
      unused_params: loss specific parameters.

    Returns:
      A scalar loss tensor.
    """
    raise NotImplementedError()


class CrossEntropyLoss(BaseLoss):
  """Calculate the cross entropy loss between the predictions and labels.
  """

  def calculate_loss(self, predictions, labels, weights=None, **unused_params):
    with tf.name_scope("loss_xent"):
      epsilon = 10e-6
      if FLAGS.label_smoothing:
        float_labels = smoothing(labels)
      else:
        float_labels = tf.cast(labels, tf.float32)
      cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + (
          1 - float_labels) * tf.log(1 - predictions + epsilon)
      cross_entropy_loss = tf.negative(cross_entropy_loss)
      if weights is not None:
        print cross_entropy_loss, weights
        weighted_loss = tf.einsum("ij,i->ij", cross_entropy_loss, weights)
        print "create weighted_loss", weighted_loss
        return tf.reduce_mean(tf.reduce_sum(weighted_loss, 1))
      else:
        return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))

class HingeLoss(BaseLoss):
  """Calculate the hinge loss between the predictions and labels.

  Note the subgradient is used in the backpropagation, and thus the optimization
  may converge slower. The predictions trained by the hinge loss are between -1
  and +1.
  """

  def calculate_loss(self, predictions, labels, b=1.0, **unused_params):
    with tf.name_scope("loss_hinge"):
      float_labels = tf.cast(labels, tf.float32)
      all_zeros = tf.zeros(tf.shape(float_labels), dtype=tf.float32)
      all_ones = tf.ones(tf.shape(float_labels), dtype=tf.float32)
      sign_labels = tf.subtract(tf.scalar_mul(2, float_labels), all_ones)
      hinge_loss = tf.maximum(
          all_zeros, tf.scalar_mul(b, all_ones) - sign_labels * predictions)
      return tf.reduce_mean(tf.reduce_sum(hinge_loss, 1))

class SoftmaxLoss(BaseLoss):
  """Calculate the softmax loss between the predictions and labels.

  The function calculates the loss in the following way: first we feed the
  predictions to the softmax activation function and then we calculate
  the minus linear dot product between the logged softmax activations and the
  normalized ground truth label.

  It is an extension to the one-hot label. It allows for more than one positive
  labels for each sample.
  """

  def calculate_loss(self, predictions, labels, **unused_params):
    with tf.name_scope("loss_softmax"):
      epsilon = 10e-8
      float_labels = tf.cast(labels, tf.float32)
      # l1 normalization (labels are no less than 0)
      label_rowsum = tf.maximum(
          tf.reduce_sum(float_labels, 1, keep_dims=True),
          epsilon)
      norm_float_labels = tf.div(float_labels, label_rowsum)
      softmax_outputs = tf.nn.softmax(predictions)
      softmax_loss = tf.negative(tf.reduce_sum(
          tf.multiply(norm_float_labels, tf.log(softmax_outputs)), 1))
    return tf.reduce_mean(softmax_loss)

class MultiTaskLoss(BaseLoss):
  """This is a vitural loss
  """
  def calculate_loss(self, unused_predictions, unused_labels, **unused_params):
    raise NotImplementedError()

  def get_support(self, labels, support_type=None):
    if support_type == None:
      support_type = FLAGS.support_type
    if "," in support_type:
      new_labels = []
      for st in support_type.split(","):
        new_labels.append(tf.cast(self.get_support(labels, st), dtype=tf.float32))
      support_labels = tf.concat(new_labels, axis=1)
      return support_labels
    elif support_type == "vertical":
      num_classes = FLAGS.num_classes
      num_verticals = FLAGS.num_verticals
      vertical_file = FLAGS.vertical_file
      vertical_mapping = np.zeros([num_classes, num_verticals], dtype=np.float32)
      float_labels = tf.cast(labels, dtype=tf.float32)
      with open(vertical_file) as F:
        for line in F:
          group = map(int, line.strip().split())
          if len(group) == 2:
            x, y = group
            vertical_mapping[x, y] = 1
      vm_init = tf.constant_initializer(vertical_mapping)
      vm = tf.get_variable("vm", shape = [num_classes, num_verticals], 
                           trainable=False, initializer=vm_init)
      vertical_labels = tf.matmul(float_labels, vm)
      return tf.cast(vertical_labels > 0.2, tf.float32)
    elif support_type == "frequent":
      num_frequents = FLAGS.num_frequents
      frequent_labels = tf.slice(labels, begin=[0, 0], size=[-1, num_frequents])
      frequent_labels = tf.cast(frequent_labels, dtype=tf.float32)
      return frequent_labels
    elif support_type == "label":
      float_labels = tf.cast(labels, dtype=tf.float32)
      return float_labels
    else:
      raise NotImplementedError()

class MultiTaskCrossEntropyLoss(MultiTaskLoss):
  """Calculate the loss between the predictions and labels.
  """
  def calculate_loss(self, predictions, support_predictions, labels, **unused_params):
    support_labels = self.get_support(labels)
    ce_loss_fn = CrossEntropyLoss()
    cross_entropy_loss = ce_loss_fn.calculate_loss(predictions, labels, **unused_params)
    cross_entropy_loss2 = ce_loss_fn.calculate_loss(support_predictions, support_labels, **unused_params)
    return cross_entropy_loss * (1.0 - FLAGS.support_loss_percent) + cross_entropy_loss2 * FLAGS.support_loss_percent

