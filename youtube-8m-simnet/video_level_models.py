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
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 2,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")
flags.DEFINE_integer(
    "num_embeddings", 256,
    "The dimension of sample embedding space.")
flags.DEFINE_integer(
    "num_negative_samples", 10,
    "The ratio of negative samples to positive samples.")

class LinearModel(models.BaseModel):
  """ Linear model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """ Creates a linear model.
    
    Args:
        model_input: 'batch' x 'num_features' matrix of input features.
        vocab_size: The number of classes in the dataset.
    Returns:
        A dictionary with a tensor containing the vector predictions of the
        model in the 'predictions' key. The dimensions of the tensor are
        batch_size x num_embeddings. The tensor contains positive predictions
        in the 'positive' key (num_classes x num_embeddings), and negative 
        predictions in the 'negative' key (num_negative_samples x num_classes x
        num_embeddings).
    """
    num_embeddings = FLAGS.num_embeddings
    num_negative_samples = FLAGS.num_negative_samples 
    with tf.variable_scope("linear"):
      predictions = slim.fully_connected(
          model_input, num_embeddings, activation_fn=None,
          weights_initializer=tf.truncated_normal_initializer(mean=0, stddev=1.0),
          weights_regularizer=slim.l2_regularizer(l2_penalty))
      label_embeddings = tf.get_variable("pos", shape = [vocab_size, num_embeddings],
          dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0, stddev=1.0),
          regularizer=slim.l2_regularizer(l2_penalty))
      positives = label_embeddings
      negative_samples = tf.random_uniform([num_negative_samples, vocab_size], 
          minval=0, maxval=vocab_size, dtype=tf.int32)
      negatives = tf.nn.embedding_lookup(label_embeddings, negative_samples)
    return {"predictions": predictions, "positives": positives, "negatives": negatives}
  
class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   sub_scope="",
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates"+sub_scope)
    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts"+sub_scope)

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}
