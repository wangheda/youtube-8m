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

"""Contains model definitions."""
import math

import models
import tensorflow as tf

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "hidden_size_1", 100,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")
flags.DEFINE_integer(
    "hidden_size_2", 25,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")
flags.DEFINE_integer(
    "hidden_channels", 3,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")

class AutoEncoderSoftmaxModel(models.BaseModel):
    """Logistic model with L2 regularization."""

    def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
        """Creates a logistic model.

        Args:
          model_input: 'batch' x 'num_features' matrix of input features.
          vocab_size: The number of classes in the dataset.

        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes."""
        model_input = tf.cast(model_input,dtype=tf.float32)
        hidden_size_1 = FLAGS.hidden_size_1
        hidden_size_2 = FLAGS.hidden_size_2
        hidden_channels = FLAGS.hidden_channels
        with tf.name_scope("autoencoder"):
            hidden_1 = slim.fully_connected(
                model_input,
                hidden_size_1*hidden_channels,
                activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="hidden_1")
            hidden_1 = tf.reshape(hidden_1,[-1,hidden_size_1])
            hidden_2 = slim.fully_connected(
                hidden_1,
                hidden_size_2,
                activation_fn=tf.nn.softmax,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="hidden_2")
            hidden_2 = tf.reshape(hidden_2,[-1,hidden_size_2*hidden_channels])

            output_1 = slim.fully_connected(
                hidden_2,
                hidden_size_1*hidden_channels,
                activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="output_1")
            output_2 = slim.fully_connected(
                output_1,
                vocab_size,
                activation_fn=tf.nn.sigmoid,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="output_2")
        loss_sparse = tf.negative(tf.reduce_mean(tf.reduce_sum(hidden_2*tf.log(hidden_2+1e-6),axis=1)))/hidden_channels
        return {"predictions": output_2, "loss_sparse": loss_sparse}

class AutoEncoderModel(models.BaseModel):
    """Logistic model with L2 regularization."""

    def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
        """Creates a logistic model.

        Args:
          model_input: 'batch' x 'num_features' matrix of input features.
          vocab_size: The number of classes in the dataset.

        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes."""
        model_input = tf.cast(model_input,dtype=tf.float32)
        hidden_size_1 = FLAGS.hidden_size_1
        hidden_size_2 = FLAGS.hidden_size_2
        with tf.name_scope("autoencoder"):
            hidden_1 = slim.fully_connected(
                model_input,
                hidden_size_1,
                activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="hidden_1")
            hidden_2 = slim.fully_connected(
                hidden_1,
                hidden_size_2,
                activation_fn=tf.nn.sigmoid,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="hidden_2")
            noise = tf.random_normal(shape=tf.shape(hidden_2), mean=0.0, stddev=0.2, dtype=tf.float32)
            hidden_noise = 1.0-tf.nn.relu(1.0-tf.nn.relu(hidden_2 + noise))
            output_1 = slim.fully_connected(
                hidden_noise,
                hidden_size_1,
                activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="output_1")
            output_2 = slim.fully_connected(
                output_1,
                vocab_size,
                activation_fn=tf.nn.sigmoid,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="output_2")
        #loss_sparse = 10*tf.negative(tf.reduce_mean(hidden_2*tf.log(hidden_2+1e-6)+(1.0-hidden_2)*tf.log(1.0-hidden_2+1e-6)))
        loss_sparse = 0.0
        return {"predictions": output_2, "loss_sparse": loss_sparse}