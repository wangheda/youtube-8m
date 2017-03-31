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
import numpy as np
FLAGS = flags.FLAGS
flags.DEFINE_integer(
  "num_pairs", 10,
  "The number of pairs (excluding the dummy 'expert') used for Hingeloss.")
flags.DEFINE_integer(
  "class_num", 25,
  "The number of pairs (excluding the dummy 'expert') used for Hingeloss.")
flags.DEFINE_string("class_file", "./resources/labels_class.out",
                    "The directory to save the model files in.")

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

  def calculate_loss(self, predictions, labels, **unused_params):
    with tf.name_scope("loss_xent"):
      epsilon = 10e-6
      float_labels = tf.cast(labels, tf.float32)
      cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + (
            1 - float_labels) * tf.log(1 - predictions + epsilon)
      cross_entropy_loss = tf.negative(cross_entropy_loss)
      return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))

  def calculate_loss_mix(self, predictions, predictions_class, labels, **unused_params):
    with tf.name_scope("loss_xent"):
      epsilon = 10e-6
      float_labels = tf.cast(labels, tf.float32)
      cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + (
             1 - float_labels) * tf.log(1 - predictions + epsilon)
      seq = np.loadtxt(FLAGS.class_file)
      tf_seq = tf.one_hot(tf.constant(seq,dtype=tf.int32),FLAGS.class_num)
      float_classes = tf.matmul(float_labels,tf_seq)
      class_true = tf.ones(tf.shape(float_classes))
      class_false = tf.zeros(tf.shape(float_classes))
      float_classes = tf.where(tf.greater(float_classes, class_false), class_true, class_false)
      cross_entropy_class = float_classes * tf.log(predictions_class + epsilon) + (
             1 - float_classes) * tf.log(1 - predictions_class + epsilon)
      cross_entropy_loss = tf.negative(cross_entropy_loss)
      cross_entropy_class = tf.negative(cross_entropy_class)
      return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1)) + tf.reduce_mean(tf.reduce_sum(cross_entropy_class, 1))

class CrossEntropyLoss_weight(BaseLoss):
  """Calculate the cross entropy loss between the predictions and labels.
  """

  def calculate_loss(self, predictions, labels, **unused_params):
    with tf.name_scope("loss_xent"):
      epsilon = 10e-6
      vocab_size = predictions.get_shape().as_list()[1]
      float_labels = tf.cast(labels, tf.float32)
      cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + (
          1 - float_labels) * tf.log(1 - predictions + epsilon)
      cross_entropy_loss = tf.negative(cross_entropy_loss)
      neg_labels = 1 - float_labels
      predictions_pos = predictions*float_labels+10*neg_labels
      predictions_minpos = tf.reduce_min(predictions_pos,axis=1,keep_dims=True)
      predictions_neg = predictions*neg_labels-10*float_labels
      predictions_maxneg = tf.reduce_max(predictions_neg,axis=1,keep_dims=True)
      mask_1 = tf.cast(tf.greater_equal(predictions_neg, predictions_minpos),dtype=tf.float32)
      mask_2 = tf.cast(tf.less_equal(predictions_pos, predictions_maxneg),dtype=tf.float32)
      cross_entropy_loss = cross_entropy_loss*(mask_1+mask_2)*10 + cross_entropy_loss
      return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))


class HingeLoss_cos(BaseLoss):
  """Calculate the hinge loss between the predictions and labels.

  Note the subgradient is used in the backpropagation, and thus the optimization
  may converge slower. The predictions trained by the hinge loss are between -1
  and +1.
  """
  def calculate_loss(self, predictions, labels, b1=1.6, b2=-0.4, **unused_params):
    with tf.name_scope("loss_hinge"):
      float_labels = tf.cast(labels, tf.float32)
      neg_labels = 1 - float_labels
      shape = predictions.get_shape().as_list()[1]
      all_zeros = tf.zeros([FLAGS.batch_size,shape,FLAGS.num_pairs], dtype=tf.float32)

      """
      predictions_minpos = predictions*float_labels+10*neg_labels
      predictions_minpos = tf.reduce_min(predictions_minpos,axis=1,keep_dims=True)
      predictions_maxneg = predictions*neg_labels-10*float_labels
      mask_1 = tf.cast(tf.greater_equal(predictions_maxneg, predictions_minpos),dtype=tf.float32)
      mask_2 = neg_labels
      pos_indices = []
      for i in range(FLAGS.batch_size):
        pos_var = tf.where(tf.not_equal(float_labels[i,:],all_zeros[i,:]))
        pos_indice = tf.tile(tf.reshape(pos_var[:,0],[-1]),[shape])+i*shape
        pos_indice = pos_indice[0:shape]
        rand_var = tf.random_uniform([shape],0,shape, dtype=tf.int32)
        pos_indice = tf.gather(pos_indice,rand_var)
        pos_indices.append(pos_indice)

      pos_indices = tf.reshape(tf.stack(pos_indices,axis=0), [-1])

      predictions_shuffle = tf.reshape(tf.gather(tf.reshape(predictions, [-1]),pos_indices), [-1,shape])
      hinge_loss = tf.maximum(all_zeros, b1 - predictions_shuffle + predictions)
      hinge_loss = hinge_loss*mask_1 + hinge_loss*mask_2
      return tf.reduce_mean(tf.reduce_sum(hinge_loss,axis=1))"""

      predictions_shuffles = []
      predictions_pos = tf.reshape(predictions*float_labels+(2-predictions)*neg_labels,[-1,shape,1])
      predictions_neg = predictions*neg_labels+(2-predictions)*float_labels
      predictions_org = tf.reshape(predictions,[-1,shape,1])
      for i in range(FLAGS.num_pairs):
        rand_var = tf.random_uniform([FLAGS.batch_size, shape],0,shape, dtype=tf.int32)+tf.reshape(tf.range(0, FLAGS.batch_size) * shape,[FLAGS.batch_size,1])
        neg_indice = tf.reshape(rand_var, [-1])
        predictions_shuffle = tf.reshape(tf.gather(tf.reshape(predictions_neg,[-1]),neg_indice),[-1,shape])
        predictions_shuffles.append(tf.reshape(predictions_shuffle,[-1,shape]))

      predictions_shuffles = tf.stack(predictions_shuffles,axis=2)

      const = tf.reshape(b1*float_labels+b2*neg_labels,[-1,shape,1])
      hinge_loss = tf.maximum(all_zeros, b1 - predictions_pos + predictions_shuffles)
      return tf.reduce_mean(tf.reduce_sum(hinge_loss,axis=1))


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
