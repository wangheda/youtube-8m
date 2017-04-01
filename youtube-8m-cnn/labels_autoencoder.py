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
            output_1 = slim.fully_connected(
                hidden_2,
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
            return {"predictions": output_2}
    def get_forward_parameters(self):
        t_vars = tf.trainable_variables()
        h1_vars_weight = [var for var in t_vars if 'hidden_1' in var.name and 'weights' in var.name]
        h1_vars_biases = [var for var in t_vars if 'hidden_1' in var.name and 'biases' in var.name]
        h2_vars_weight = [var for var in t_vars if 'hidden_2' in var.name and 'weights' in var.name]
        h2_vars_biases = [var for var in t_vars if 'hidden_2' in var.name and 'biases' in var.name]
        h1_vars_biases = tf.reshape(h1_vars_biases[0],[1,FLAGS.hidden_size_1])
        h2_vars_biases = tf.reshape(h2_vars_biases[0],[1,FLAGS.hidden_size_2])
        vars_1 = tf.concat((h1_vars_weight[0],h1_vars_biases),axis=0)
        vars_2 = tf.concat((h2_vars_weight[0],h2_vars_biases),axis=0)
        return [vars_1,vars_2]