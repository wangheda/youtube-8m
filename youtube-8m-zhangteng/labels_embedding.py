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
    "hidden_size", 1024,
    "The size of hidden layers.")
flags.DEFINE_integer(
    "top_k", 30,
    "The maximum number of positive labels in one sample.")

class EmbeddingSigmoidModel(models.BaseModel):
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
        hidden_size = FLAGS.hidden_size

        model_mask, indices_input = tf.nn.top_k(model_input, k=FLAGS.top_k)
        indices_input = tf.reshape(indices_input, [-1])
        models_mask = tf.reshape(model_mask, [-1,FLAGS.top_k,1])
        with tf.name_scope("embedding"):
            embeddings = tf.Variable(
                tf.random_uniform([vocab_size, hidden_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, indices_input)
            output = slim.fully_connected(
                embed,
                vocab_size,
                activation_fn=tf.nn.sigmoid,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="output")
        indices_one_hot = tf.one_hot(indices_input, vocab_size)
        output = output * (1 - indices_one_hot) + indices_one_hot
        output_val = tf.reshape(output,[-1,FLAGS.top_k,vocab_size])
        predictions_val = tf.reduce_sum(output_val*models_mask, axis=1)/tf.reduce_sum(models_mask, axis=1)
        return {"predictions": output, "predictions_val": predictions_val}
