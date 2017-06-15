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
flags.DEFINE_string("class_file", "./resources/labels_knowledge.out",
                    "The directory to save the 24 top-level verticals in, used for 'calculate_loss_mix'")
flags.DEFINE_string("frequent_file", "./resources/labels_frequent.out",
                    "The directory to save the frequency of 4716 labels in, used only in early experiment.")
flags.DEFINE_string("autoencoder_dir", "./resources/",
                    "The directory to save the autoencoder model layers in.")
flags.DEFINE_string("support_type", None,
                    "The support type for mix models, options are None, class, frequent and encoder,"
                    "used for 'calculate_loss_mix'.")
flags.DEFINE_string("loss_function", None,
                    "different loss funtions used in CrossEntropyLoss.")
flags.DEFINE_integer("encoder_layers", 2,
                     "The number of autoencoder layers.")
flags.DEFINE_float("jsd_pi", 0.5,
                   "wight used when loss function is loss_jsd.")
flags.DEFINE_float("threshold", 0.5,
                   "used only in early experiment.")

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
            origin_labels = tf.cast(labels, tf.float32)
            vocab_size = origin_labels.get_shape().as_list()[1]
            float_labels = tf.tile(tf.reshape(origin_labels,[-1, 1, vocab_size]),[1,FLAGS.top_k,1])
            float_labels = tf.reshape(float_labels,[-1,vocab_size])
            cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + (
                1 - float_labels) * tf.log(1 - predictions + epsilon)
            cross_entropy_loss = tf.negative(cross_entropy_loss)
            num_labels = tf.minimum(tf.reduce_sum(origin_labels,axis=1),tf.constant(FLAGS.top_k,dtype=tf.float32))
            mask = tf.reshape(tf.sequence_mask(num_labels,tf.constant(FLAGS.top_k,dtype=tf.float32),dtype=tf.float32),[-1])
            cross_entropy_loss = tf.reduce_sum(tf.reduce_sum(cross_entropy_loss, 1)*mask)/(tf.reduce_sum(mask)+epsilon)

            return cross_entropy_loss


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
        bound = FLAGS.softmax_bound
        vocab_size_1 = bound
        with tf.name_scope("loss_softmax"):
            epsilon = 10e-8
            float_labels = tf.cast(labels, tf.float32)
            labels_1 = float_labels[:,:vocab_size_1]
            predictions_1 = predictions[:,:vocab_size_1]
            cross_entropy_loss = CrossEntropyLoss().calculate_loss(predictions_1,labels_1)
            lables_2 = float_labels[:,vocab_size_1:]
            predictions_2 = predictions[:,vocab_size_1:]
            # l1 normalization (labels are no less than 0)
            label_rowsum = tf.maximum(
                tf.reduce_sum(lables_2, 1, keep_dims=True),
                epsilon)
            label_append = 1.0-tf.reduce_max(lables_2, 1, keep_dims=True)
            norm_float_labels = tf.concat((tf.div(lables_2, label_rowsum),label_append),axis=1)
            predictions_append = 1.0-tf.reduce_sum(predictions_2, 1, keep_dims=True)
            softmax_outputs = tf.concat((predictions_2,predictions_append),axis=1)
            softmax_loss = norm_float_labels * tf.log(softmax_outputs + epsilon) + (
                                                                                       1 - norm_float_labels) * tf.log(1 - softmax_outputs + epsilon)
            softmax_loss = tf.negative(tf.reduce_sum(softmax_loss, 1))
        return tf.reduce_mean(softmax_loss) + cross_entropy_loss

    def calculate_loss_mix(self, predictions, predictions_class, labels, **unused_params):
        with tf.name_scope("loss_softmax_mix"):
            vocab_size = labels.get_shape().as_list()[1]
            cross_entropy_class = tf.constant(0.0)
            for i in range(FLAGS.moe_layers):
                predictions_subclass = predictions_class[:,i*vocab_size:(i+1)*vocab_size]
                cross_entropy_class = cross_entropy_class + self.calculate_loss(predictions_subclass,labels)
            cross_entropy_loss = self.calculate_loss(predictions,labels)
            return cross_entropy_loss + 0.1*cross_entropy_class
