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

import tensorflow as tf
from tensorflow import flags

FLAGS = flags.FLAGS
flags.DEFINE_float("false_negative_punishment", 1.0, 
                   "punishment constant to 1 classified to 0")
flags.DEFINE_float("false_positive_punishment", 1.0, 
                   "punishment constant to 0 classified to 1")
flags.DEFINE_integer("num_classes", 4716,
                   "number of classes")

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


class WeightedCrossEntropyLoss(BaseLoss):
  """Calculate the cross entropy loss between the predictions and labels.
     1 -> 0 will be punished hard, while the other way will not punished not hard.
  """

  def calculate_loss(self, predictions, labels, **unused_params):
    false_positive_punishment = FLAGS.false_positive_punishment
    false_negative_punishment = FLAGS.false_negative_punishment
    with tf.name_scope("loss_xent_recall"):
      epsilon = 10e-6
      float_labels = tf.cast(labels, tf.float32)
      cross_entropy_loss = false_negative_punishment * float_labels * tf.log(predictions + epsilon) \
          + false_positive_punishment * ( 1 - float_labels) * tf.log(1 - predictions + epsilon)
      cross_entropy_loss = tf.negative(cross_entropy_loss)
      return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))


class CrossEntropyLoss(BaseLoss):
  """Calculate the cross entropy loss between the predictions and labels.
  """

  def calculate_loss(self, predictions, labels, **unused_params):
    with tf.name_scope("loss_xent"):
      epsilon = 10e-6
      float_labels = tf.cast(labels, tf.float32)
      cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + (
          1 - float_labels) * tf.log(1 - predictions + epsilon)
      cross_entropy_loss = tf.negative(cross_entropy_loss)
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

class PairwiseHingeLoss(BaseLoss):
  def calculate_loss(self, predictions, labels, margin=0.2, adaptive=3.0, origin=1.0, **unused_params):
    batch_size = FLAGS.batch_size
    num_classes = FLAGS.num_classes
    with tf.name_scope("loss_hinge"):
      # get sim_neg
      mask = tf.cast(labels, tf.float32)
      reverse_mask = 1.0 - mask
      min_true_pred = tf.reduce_min((predictions - 1.0) * mask, axis=1, keep_dims=True) + 1.0
      mask_wrong = tf.stop_gradient(tf.cast(predictions > (min_true_pred - margin), tf.float32) * reverse_mask)
      # get positve samples
      int_labels = tf.cast(labels, tf.int32)
      sample_labels = tf.unstack(int_labels, num=batch_size, axis=0)
      sample_predictions = tf.unstack(predictions, num=batch_size, axis=0)
      positive_predictions = []
      for sample_label, sample_prediction in zip(sample_labels, sample_predictions):
        indices = tf.where(sample_label > 0)
        expanded_indices = tf.tile(indices[:,0], [num_classes])[:num_classes]
        rand_arrange = tf.random_uniform([num_classes], minval=0, maxval=num_classes, dtype=tf.int32)
        positive_indices = tf.stop_gradient(tf.gather(expanded_indices, rand_arrange))
        positive_prediction = tf.gather(sample_prediction, positive_indices)
        positive_predictions.append(positive_prediction)
      positive_predictions = tf.stack(positive_predictions)
      # hinge_loss
      hinge_loss = tf.maximum(predictions - positive_predictions + margin, 0.0)
      adaptive_loss = hinge_loss * mask_wrong
      adaptive_loss = tf.reduce_mean(tf.reduce_sum(adaptive_loss, axis=1))
      origin_loss = hinge_loss * reverse_mask
      origin_loss = tf.reduce_mean(tf.reduce_sum(origin_loss, axis=1))
      loss = adaptive * adaptive_loss + origin * origin_loss
      return loss

class MixedLoss(BaseLoss):
  def calculate_loss(self, predictions, labels, margin=0.2, adaptive=3, **unused_params):
    cross_ent_loss = CrossEntropyLoss()
    pairwise_loss = PairwiseHingeLoss()
    ce_loss = cross_ent_loss.calculate_loss(predictions, labels, **unused_params)
    pw_loss = pairwise_loss.calculate_loss(predictions, labels, margin, adaptive=1.0, origin=0.0, **unused_params)
    return ce_loss + pw_loss * 0.05

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
