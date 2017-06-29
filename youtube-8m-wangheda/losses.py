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


class WeightedCrossEntropyLoss(BaseLoss):
  """Calculate the cross entropy loss between the predictions and labels.
     1 -> 0 will be punished hard, while the other way will not punished not hard.
  """

  def calculate_loss(self, predictions, labels, **unused_params):
    false_positive_punishment = FLAGS.false_positive_punishment
    false_negative_punishment = FLAGS.false_negative_punishment
    with tf.name_scope("loss_xent_recall"):
      epsilon = 10e-6
      if FLAGS.label_smoothing:
        float_labels = smoothing(labels)
      else:
        float_labels = tf.cast(labels, tf.float32)
      cross_entropy_loss = false_negative_punishment * float_labels * tf.log(predictions + epsilon) \
          + false_positive_punishment * ( 1 - float_labels) * tf.log(1 - predictions + epsilon)
      cross_entropy_loss = tf.negative(cross_entropy_loss)
      return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))


class MeanSquareErrorLoss(BaseLoss):
  """Calculate the MSE loss between the predictions and labels.
  """

  def calculate_loss(self, predictions, labels, **unused_params):
    with tf.name_scope("loss_xent"):
      epsilon = 10e-6
      if FLAGS.label_smoothing:
        float_labels = smoothing(labels)
      else:
        float_labels = tf.cast(labels, tf.float32)
      mse_loss = tf.square(float_labels - predictions)
      return tf.reduce_mean(tf.reduce_sum(mse_loss, 1))

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


class MultiTaskCrossEntropyAndSoftmaxLoss(MultiTaskLoss):
  """Calculate the loss between the predictions and labels.
  """
  def calculate_loss(self, predictions, support_predictions, labels, **unused_params):
    support_labels = self.get_support(labels)
    ce_loss_fn = CrossEntropyLoss()
    cross_entropy_loss = ce_loss_fn.calculate_loss(predictions, labels, **unused_params)
    sm_loss_fn = SoftmaxLoss()
    softmax_loss = sm_loss_fn.calculate_loss(support_predictions, support_labels, **unused_params)
    return cross_entropy_loss * (1.0 - FLAGS.support_loss_percent) + softmax_loss * FLAGS.support_loss_percent

class MultiTaskCrossEntropyLoss(MultiTaskLoss):
  """Calculate the loss between the predictions and labels.
  """
  def calculate_loss(self, predictions, support_predictions, labels, **unused_params):
    support_labels = self.get_support(labels)
    ce_loss_fn = CrossEntropyLoss()
    cross_entropy_loss = ce_loss_fn.calculate_loss(predictions, labels, **unused_params)
    cross_entropy_loss2 = ce_loss_fn.calculate_loss(support_predictions, support_labels, **unused_params)
    return cross_entropy_loss * (1.0 - FLAGS.support_loss_percent) + cross_entropy_loss2 * FLAGS.support_loss_percent

class BatchAgreementCrossEntropyLoss(BaseLoss):
  """loss that exagerate those points that break the batch-wise order
  """
  def calculate_loss(self, predictions, labels, **unused_params):
    with tf.name_scope("loss_xent_batch"):
      batch_agreement = FLAGS.batch_agreement
      epsilon = 10e-6
      float_batch_size = float(FLAGS.batch_size)

      float_labels = tf.cast(labels, tf.float32)
      cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + (
          1 - float_labels) * tf.log(1 - predictions + epsilon)
      cross_entropy_loss = tf.negative(cross_entropy_loss)
      
      positive_predictions = predictions * float_labels + 1.0 - float_labels
      min_pp = tf.reduce_min(positive_predictions)

      negative_predictions = predictions * (1.0 - float_labels)
      max_np = tf.reduce_max(negative_predictions)

      # 1s that fall under 0s
      false_negatives = tf.cast(predictions < max_np, tf.float32) * float_labels
      num_fn = tf.reduce_sum(false_negatives)
      center_fn = tf.reduce_sum(predictions * false_negatives) / num_fn

      # 0s that grow over 1s
      false_positives = tf.cast(predictions > min_pp, tf.float32) * (1.0 - float_labels)
      num_fp = tf.reduce_sum(false_positives)
      center_fp = tf.reduce_sum(predictions * false_positives) / num_fp

      false_range = tf.maximum(epsilon, max_np - min_pp)

      # for 1s that fall under 0s
      weight_fn = tf.nn.sigmoid((center_fp - predictions) / false_range * 3.0) * (num_fp / float_batch_size) * false_negatives
      # for 0s that grow over 1s
      weight_fp = tf.nn.sigmoid((predictions - center_fn) / false_range * 3.0) * (num_fn / float_batch_size) * false_positives
      
      weight = (weight_fn + weight_fp) * batch_agreement + 1.0
      print weight
      return tf.reduce_mean(tf.reduce_sum(weight * cross_entropy_loss, 1))

class TopKBatchAgreementCrossEntropyLoss(BaseLoss):
  """loss that exagerate those points that break the batch-wise order
  """
  def calculate_loss(self, predictions, labels, topk=20, **unused_params):
    with tf.name_scope("loss_xent_batch"):
      batch_agreement = FLAGS.batch_agreement
      epsilon = 10e-6
      float_batch_size = float(FLAGS.batch_size)

      topk_predictions, _ = tf.nn.top_k(predictions, k=20)
      min_topk_predictions = tf.reduce_min(topk_predictions, axis=1, keep_dims=True)
      topk_mask = tf.cast(predictions >= min_topk_predictions, dtype=tf.float32)

      float_labels = tf.cast(labels, tf.float32)
      cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + (
          1 - float_labels) * tf.log(1 - predictions + epsilon)
      cross_entropy_loss = tf.negative(cross_entropy_loss)
      
      # minimum positive predictions in topk
      positive_predictions = (predictions * float_labels * topk_mask) + 1.0 - (float_labels * topk_mask)
      min_pp = tf.reduce_min(positive_predictions)

      # maximum negative predictions
      negative_predictions = predictions * (1.0 - float_labels)
      max_np = tf.reduce_max(negative_predictions)

      # 1s that fall under top-k
      false_negatives = tf.cast(predictions < min_topk_predictions, tf.float32) * float_labels
      # 0s that grow over 1s in top-k
      false_positives = tf.cast(predictions > min_pp, tf.float32) * (1.0 - float_labels) * topk_mask

      weight = (false_negatives + false_positives) * batch_agreement + 1.0
      weight = tf.stop_gradient(weight)
      print weight
      return tf.reduce_mean(tf.reduce_sum(weight * cross_entropy_loss, 1))

class MultiTaskDivergenceCrossEntropyLoss(MultiTaskLoss):
  """Calculate the loss between the predictions and labels.
  """
  def calculate_loss(self, predictions, support_predictions, labels, **unused_params):
    """ 
    support_predictions batch_size x num_models x num_classes
    predictions = tf.reduce_mean(support_predictions, axis=1)
    """
    model_count = tf.shape(support_predictions)[1]
    vocab_size = tf.shape(support_predictions)[2]

    mean_predictions = tf.reduce_mean(support_predictions, axis=1, keep_dims=True)
    support_labels = tf.tile(tf.expand_dims(tf.cast(labels, dtype=tf.float32), axis=1), multiples=[1,model_count,1])
    support_means = tf.stop_gradient(tf.tile(mean_predictions, multiples=[1,model_count,1]))

    support_predictions = tf.reshape(support_predictions, shape=[-1,model_count*vocab_size])
    support_labels = tf.reshape(support_labels, shape=[-1,model_count*vocab_size])
    support_means = tf.reshape(support_means, shape=[-1,model_count*vocab_size])

    ce_loss_fn = CrossEntropyLoss()
    # The cross entropy between predictions and ground truth
    cross_entropy_loss = ce_loss_fn.calculate_loss(support_predictions, support_labels, **unused_params)
    # The cross entropy between predictions and mean predictions
    divergence = ce_loss_fn.calculate_loss(support_predictions, support_means, **unused_params)

    loss = cross_entropy_loss * (1.0 - FLAGS.support_loss_percent) - divergence * FLAGS.support_loss_percent
    return loss

class MultiTaskDivergenceCrossEntropyAndMSELoss(MultiTaskLoss):
  """Calculate the loss between the predictions and labels.
  """
  def calculate_loss(self, predictions, support_predictions, labels, **unused_params):
    """ 
    support_predictions batch_size x num_models x num_classes
    predictions = tf.reduce_mean(support_predictions, axis=1)
    """
    model_count = tf.shape(support_predictions)[1]
    vocab_size = tf.shape(support_predictions)[2]

    mean_predictions = tf.reduce_mean(support_predictions, axis=1, keep_dims=True)
    support_labels = tf.tile(tf.expand_dims(tf.cast(labels, dtype=tf.float32), axis=1), multiples=[1,model_count,1])
    support_means = tf.stop_gradient(tf.tile(mean_predictions, multiples=[1,model_count,1]))

    support_predictions = tf.reshape(support_predictions, shape=[-1,model_count*vocab_size])
    support_labels = tf.reshape(support_labels, shape=[-1,model_count*vocab_size])
    support_means = tf.reshape(support_means, shape=[-1,model_count*vocab_size])

    ce_loss_fn = CrossEntropyLoss()
    # The cross entropy between predictions and ground truth
    cross_entropy_loss = ce_loss_fn.calculate_loss(support_predictions, support_labels, **unused_params)

    mse_loss_fn = MeanSquareErrorLoss()
    # The square error between predictions and mean predictions
    divergence = mse_loss_fn.calculate_loss(support_predictions, support_means, **unused_params)

    loss = cross_entropy_loss * (1.0 - FLAGS.support_loss_percent) - divergence * FLAGS.support_loss_percent
    return loss

