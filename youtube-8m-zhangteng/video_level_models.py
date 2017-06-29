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
import numpy as np
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 8,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")
flags.DEFINE_integer(
    "moe_num_extend", 8,
    "The number of attention outputs, used for MoeExtendModel.")
flags.DEFINE_string("moe_method", "none",
                    "The pooling method used in the DBoF cluster layer. "
                    "used for MoeMaxModel.")
flags.DEFINE_integer(
    "class_size", 200,
    "The dimention of prediction projection, used for all chain models.")
flags.DEFINE_integer(
    "encoder_size", 100,
    "The dimention of prediction encoder, used for all mix models.")
flags.DEFINE_integer(
    "hidden_size_1", 100,
    "The size of the first hidden layer, used forAutoEncoderModel.")
flags.DEFINE_integer(
    "hidden_channels", 3,
    "The number of hidden layers, only used in early experiment.")
flags.DEFINE_integer(
    "moe_layers", 1,
    "The number of combine layers, used for combine related models.")
flags.DEFINE_integer(
    "softmax_bound", 1000,
    "The number of labels to be a group, only used for MoeSoftmaxModel and MoeDistillSplitModel.")
flags.DEFINE_bool(
    "moe_group", False,
    "Whether to split the 4716 labels into different groups, used in MoeMix4Model and MoeNoiseModel")
flags.DEFINE_float("noise_std", 0.2, "the standard deviation of noise added to the input.")
flags.DEFINE_float("ensemble_w", 1.0, "ensemble weight used in distill chain models.")


class LogisticModel(models.BaseModel):
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
    output = slim.fully_connected(
        model_input, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
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
    shape = model_input.get_shape().as_list()

    if FLAGS.frame_features:
        model_input = tf.reshape(model_input,[-1,shape[-1]])
    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    """
    gate_w = tf.get_variable("gate_w", [shape[1], vocab_size * (num_mixtures + 1)], tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
    tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(gate_w))
    gate_activations = tf.matmul(model_input,gate_w)

    expert_w = tf.get_variable("expert_w", [shape[1], vocab_size * num_mixtures], tf.float32,
                               initializer=tf.contrib.layers.xavier_initializer())
    tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(expert_w))
    expert_v = tf.get_variable("expert_v", [vocab_size * num_mixtures], tf.float32,
                              initializer=tf.constant_initializer(0.0))
    tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(expert_v))
    expert_activations = tf.nn.xw_plus_b(model_input,expert_w,expert_v)"""

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    probabilities_by_class_and_batch = tf.reshape(probabilities_by_class_and_batch,
                                                  [-1, vocab_size])

    final_probabilities = tf.reshape(probabilities_by_class_and_batch,
                                     [-1, vocab_size])

    return {"predictions": final_probabilities}

class MoeDistillModel(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     distill_labels=None,
                     num_mixtures=None,
                     l2_penalty=1e-8,
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
        shape = model_input.get_shape().as_list()

        if FLAGS.frame_features:
            model_input = tf.reshape(model_input,[-1,shape[-1]])
        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        probabilities_by_class_and_batch = tf.reshape(probabilities_by_class_and_batch,
                                                      [-1, vocab_size])

        final_sub_probabilities = tf.reshape(probabilities_by_class_and_batch,
                                         [-1, vocab_size])

        if distill_labels is not None:
            expert_gate = slim.fully_connected(
                model_input,
                vocab_size,
                activation_fn=tf.nn.sigmoid,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="expert_gate")
            expert_gate = expert_gate*0.8 + 0.1
            final_probabilities = distill_labels*(1.0-expert_gate) + final_sub_probabilities*expert_gate
            tf.summary.histogram("expert_gate/activations", expert_gate)
        else:
            final_probabilities = final_sub_probabilities
        return {"predictions": final_probabilities, "predictions_class": final_sub_probabilities}

class MoeDistillEmbeddingModel(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     distill_labels=None,
                     num_mixtures=None,
                     l2_penalty=1e-8,
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
        """
        embedding_mat = np.loadtxt("./resources/embedding_matrix.model.gz")
        embedding_mat = tf.cast(embedding_mat,dtype=tf.float32)

        bound = FLAGS.softmax_bound
        vocab_size_1 = bound

        probabilities_by_distill = distill_labels[:, :vocab_size_1]
        embedding_mat = embedding_mat[:vocab_size_1, :]
        labels_smooth = tf.matmul(probabilities_by_distill, embedding_mat)
        probabilities_by_smooth_1 = (labels_smooth[:, :vocab_size_1] - probabilities_by_distill)/tf.reduce_sum(probabilities_by_distill,axis=1,keep_dims=True)
        probabilities_by_smooth_2 = labels_smooth[:, vocab_size_1:]/tf.reduce_sum(probabilities_by_distill,axis=1,keep_dims=True)
        labels_smooth = tf.concat((probabilities_by_smooth_1, probabilities_by_smooth_2), axis=1)"""


        expert_gate = slim.fully_connected(
            distill_labels,
            1,
            activation_fn=tf.nn.sigmoid,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="expert_gate")
        #final_probabilities = tf.clip_by_value(distill_labels + labels_smooth, 0.0, 1.0)
        final_probabilities = distill_labels
        return {"predictions": final_probabilities}

class MoeDistillChainModel(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     distill_labels=None,
                     num_mixtures=None,
                     l2_penalty=1e-8,
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
        class_size = 256
        shape = model_input.get_shape().as_list()
        if distill_labels is not None:
            class_input = slim.fully_connected(
                distill_labels,
                class_size,
                activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="class_inputs")
            model_input = tf.concat((model_input,class_input),axis=1)

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        probabilities_by_class_and_batch = tf.reshape(probabilities_by_class_and_batch,
                                                      [-1, vocab_size])

        final_probabilities = tf.reshape(probabilities_by_class_and_batch,
                                         [-1, vocab_size])

        final_probabilities = final_probabilities*FLAGS.ensemble_w + distill_labels*(1.0-FLAGS.ensemble_w)

        return {"predictions": final_probabilities}

class MoeDistillChainNormModel(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     distill_labels=None,
                     num_mixtures=None,
                     l2_penalty=1e-8,
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
        class_size = 256
        model_input = tf.nn.l2_normalize(model_input,dim=1)
        if distill_labels is not None:
            class_input = slim.fully_connected(
                distill_labels,
                class_size,
                activation_fn=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="class_inputs")
            class_input = class_input/tf.reduce_sum(distill_labels,axis=1,keep_dims=True)
            class_input = tf.nn.l2_normalize(class_input,dim=1)
            model_input = tf.concat((model_input,class_input),axis=1)

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        probabilities_by_class_and_batch = tf.reshape(probabilities_by_class_and_batch,
                                                      [-1, vocab_size])

        final_probabilities = tf.reshape(probabilities_by_class_and_batch,
                                         [-1, vocab_size])

        final_probabilities = final_probabilities*FLAGS.ensemble_w + distill_labels*(1.0-FLAGS.ensemble_w)

        return {"predictions": final_probabilities}

class MoeDistillChainNorm2Model(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     distill_labels=None,
                     num_mixtures=None,
                     l2_penalty=1e-8,
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
        class_size = 256
        model_input = tf.nn.l2_normalize(model_input,dim=1)
        if distill_labels is not None:
            class_input = slim.fully_connected(
                distill_labels,
                class_size,
                activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="class_inputs")
            class_input = class_input/tf.reduce_sum(distill_labels,axis=1,keep_dims=True)
            class_input = tf.nn.l2_normalize(class_input,dim=1)
            model_input = tf.concat((model_input,class_input),axis=1)

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        probabilities_by_class_and_batch = tf.reshape(probabilities_by_class_and_batch,
                                                      [-1, vocab_size])

        final_probabilities = tf.reshape(probabilities_by_class_and_batch,
                                         [-1, vocab_size])

        final_probabilities = final_probabilities*FLAGS.ensemble_w + distill_labels*(1.0-FLAGS.ensemble_w)

        return {"predictions": final_probabilities}

class MoeDistillSplitModel(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     distill_labels=None,
                     num_mixtures=None,
                     l2_penalty=1e-8,
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
        bound = FLAGS.softmax_bound
        vocab_size_1 = bound
        class_size = 256
        probabilities_by_distill = distill_labels[:,vocab_size_1:]

        class_input = slim.fully_connected(
            probabilities_by_distill,
            class_size,
            activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="class_inputs")
        #class_input = tf.nn.l2_normalize(class_input, dim=1)
        model_input = tf.concat((model_input,class_input),axis=1)

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size_1 * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size_1 * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        probabilities_by_class_and_batch = tf.reshape(probabilities_by_class_and_batch,
                                                      [-1, vocab_size_1])

        final_probabilities = tf.concat((probabilities_by_class_and_batch, probabilities_by_distill), axis=1)

        final_probabilities = final_probabilities*FLAGS.ensemble_w + distill_labels*(1.0-FLAGS.ensemble_w)

        return {"predictions": final_probabilities}

class MoeDistillSplit2Model(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     distill_labels=None,
                     num_mixtures=None,
                     l2_penalty=1e-8,
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
        bound = FLAGS.softmax_bound
        vocab_size_1 = bound
        class_size = 256
        probabilities_by_distill = distill_labels[:,vocab_size_1:]
        probabilities_by_residual = tf.clip_by_value(1.0-tf.reduce_sum(probabilities_by_distill,axis=1,keep_dims=True), 0.0, 1.0)
        probabilities_by_distill_residual = tf.concat((probabilities_by_residual,probabilities_by_distill), axis=1)

        class_input = slim.fully_connected(
            probabilities_by_distill_residual,
            class_size,
            activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="class_inputs")
        class_input = tf.nn.l2_normalize(class_input, dim=1)
        model_input = tf.concat((model_input,class_input),axis=1)

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size_1 * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size_1 * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        probabilities_by_class_and_batch = tf.reshape(probabilities_by_class_and_batch,
                                                      [-1, vocab_size_1])

        final_probabilities = tf.concat((probabilities_by_class_and_batch, probabilities_by_distill), axis=1)

        final_probabilities = final_probabilities*FLAGS.ensemble_w + distill_labels*(1.0-FLAGS.ensemble_w)

        return {"predictions": final_probabilities}

class MoeDistillSplit3Model(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     distill_labels=None,
                     num_mixtures=None,
                     l2_penalty=1e-8,
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
        bound = FLAGS.softmax_bound
        vocab_size_1 = bound
        vocab_size_2 = vocab_size - vocab_size_1
        class_size = 256
        probabilities_by_distill = distill_labels[:,:vocab_size_1]
        probabilities_by_residual = distill_labels[:,vocab_size_1:]
        feature_size = model_input.get_shape().as_list()[1]

        model_input = slim.fully_connected(
            model_input,
            feature_size,
            activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="model_inputs")
        model_input = tf.nn.l2_normalize(model_input, dim=1)

        gate_activations_1 = slim.fully_connected(
            model_input,
            vocab_size_1 * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates-1")
        expert_activations_1 = slim.fully_connected(
            model_input,
            vocab_size_1 * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts-1")
        gating_distribution_1 = tf.nn.softmax(tf.reshape(
            gate_activations_1,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution_1 = tf.nn.sigmoid(tf.reshape(
            expert_activations_1,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        probabilities_by_class_and_batch_1 = tf.reduce_sum(
            gating_distribution_1[:, :num_mixtures] * expert_distribution_1, 1)
        probabilities_by_class_and_batch_1 = tf.reshape(probabilities_by_class_and_batch_1,
                                                      [-1, vocab_size_1])

        probabilities_by_class = tf.concat((probabilities_by_class_and_batch_1, probabilities_by_residual), axis=1)

        class_input = slim.fully_connected(
            probabilities_by_distill,
            class_size,
            activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="class_inputs")
        class_input = tf.nn.l2_normalize(class_input, dim=1)
        model_input = tf.concat((model_input,class_input),axis=1)

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size_2 * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size_2 * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        probabilities_by_class_and_batch = tf.reshape(probabilities_by_class_and_batch,
                                                      [-1, vocab_size_2])

        final_probabilities = tf.concat((probabilities_by_distill, probabilities_by_class_and_batch), axis=1)

        final_probabilities = final_probabilities*FLAGS.ensemble_w + distill_labels*(1.0-FLAGS.ensemble_w)

        return {"predictions": final_probabilities, "predictions_class": probabilities_by_class}

class MoeDistillSplit4Model(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     distill_labels=None,
                     num_mixtures=None,
                     l2_penalty=1e-8,
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
        bound = FLAGS.softmax_bound
        vocab_size_1 = bound
        vocab_size_2 = vocab_size - vocab_size_1
        class_size = 256
        probabilities_by_distill = distill_labels[:,:vocab_size_1]
        probabilities_by_residual = distill_labels[:,vocab_size_1:]

        gate_activations_1 = slim.fully_connected(
            model_input,
            vocab_size_1 * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates-1")
        expert_activations_1 = slim.fully_connected(
            model_input,
            vocab_size_1 * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts-1")
        gating_distribution_1 = tf.nn.softmax(tf.reshape(
            gate_activations_1,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution_1 = tf.nn.sigmoid(tf.reshape(
            expert_activations_1,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        probabilities_by_class_and_batch_1 = tf.reduce_sum(
            gating_distribution_1[:, :num_mixtures] * expert_distribution_1, 1)
        probabilities_by_class_and_batch_1 = tf.reshape(probabilities_by_class_and_batch_1,
                                                        [-1, vocab_size_1])

        probabilities_by_class = tf.concat((probabilities_by_class_and_batch_1, probabilities_by_residual), axis=1)

        class_input = slim.fully_connected(
            probabilities_by_distill,
            class_size,
            activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="class_inputs")
        model_input = tf.concat((model_input,class_input),axis=1)

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size_2 * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size_2 * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        probabilities_by_class_and_batch = tf.reshape(probabilities_by_class_and_batch,
                                                      [-1, vocab_size_2])

        final_probabilities = tf.concat((probabilities_by_distill, probabilities_by_class_and_batch), axis=1)

        final_probabilities = final_probabilities*FLAGS.ensemble_w + distill_labels*(1.0-FLAGS.ensemble_w)

        return {"predictions": final_probabilities, "predictions_class": probabilities_by_class}

class MoeSoftmaxModel(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""
    def sub_model(self,
                  model_input,
                  vocab_size,
                  num_mixtures=None,
                  l2_penalty=1e-8,
                  name="",
                  **unused_params):

        num_mixtures = num_mixtures or FLAGS.moe_num_mixtures
        class_size = FLAGS.class_size
        bound = FLAGS.softmax_bound
        vocab_size_1 = bound

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size_1 * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates"+name)
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size_1 * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts"+name)

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        probabilities_by_sigmoid = tf.reshape(probabilities_by_class_and_batch,
                                                      [-1, vocab_size_1])

        vocab_size_2 = vocab_size - bound
        class_size = vocab_size_2
        channels = 1
        probabilities_by_softmax = []
        for i in range(channels):
            if i<channels-1:
                sub_vocab_size = class_size + 1
            else:
                sub_vocab_size = vocab_size_2 - (channels-1)*class_size + 1
            gate_activations = slim.fully_connected(
                model_input,
                sub_vocab_size * (num_mixtures + 1),
                activation_fn=None,
                biases_initializer=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="class_gates-%s" % i + name)
            expert_activations = slim.fully_connected(
                model_input,
                sub_vocab_size * num_mixtures,
                activation_fn=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="class_experts-%s" % i + name)

            gating_distribution = tf.nn.softmax(tf.reshape(
                gate_activations,
                [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
            expert_distribution = tf.nn.softmax(tf.reshape(
                expert_activations,
                [-1, sub_vocab_size, num_mixtures]),dim=1)  # (Batch * #Labels) x num_mixtures

            expert_distribution = tf.reshape(expert_distribution,[-1,num_mixtures])
            probabilities_by_subvocab = tf.reduce_sum(
                gating_distribution[:, :num_mixtures] * expert_distribution, 1)
            probabilities_by_subvocab = tf.reshape(probabilities_by_subvocab,
                                                   [-1, sub_vocab_size])
            probabilities_by_subvocab = probabilities_by_subvocab/tf.reduce_sum(probabilities_by_subvocab,axis=1,keep_dims=True)
            if i==0:
                probabilities_by_softmax = probabilities_by_subvocab[:,:-1]
            else:
                probabilities_by_softmax = tf.concat((probabilities_by_softmax, probabilities_by_subvocab[:,:-1]),axis=1)

        probabilities_by_class = tf.concat((probabilities_by_sigmoid,probabilities_by_softmax),axis=1)
        return probabilities_by_class

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_mixtures=None,
                     l2_penalty=1e-8,
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
        shape = model_input.get_shape().as_list()[1]
        class_size = FLAGS.class_size

        probabilities_by_class = self.sub_model(model_input,vocab_size,name="pre")
        probabilities_by_vocab = probabilities_by_class
        vocab_input = model_input
        for i in range(FLAGS.moe_layers):
            class_input_1 = slim.fully_connected(
                probabilities_by_vocab,
                class_size,
                activation_fn=tf.nn.elu,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="class_inputs1-%s" % i)
            class_input_2 = slim.fully_connected(
                1-probabilities_by_vocab,
                class_size,
                activation_fn=tf.nn.elu,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="class_inputs2-%s" % i)
            class_input_1 = tf.nn.l2_normalize(class_input_1,dim=1)*tf.sqrt(tf.cast(class_size,dtype=tf.float32)/shape)
            class_input_2 = tf.nn.l2_normalize(class_input_2,dim=1)*tf.sqrt(tf.cast(class_size,dtype=tf.float32)/shape)

            vocab_input = tf.concat((vocab_input,class_input_1,class_input_2),axis=1)

            probabilities_by_vocab = self.sub_model(vocab_input,vocab_size,name="-%s" % i)

            if i<FLAGS.moe_layers-1:
                probabilities_by_class = tf.concat((probabilities_by_class,probabilities_by_vocab),axis=1)

        final_probabilities = probabilities_by_vocab

        return {"predictions": final_probabilities, "predictions_class": probabilities_by_class}


class MoeNegativeModel(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_mixtures=None,
                     l2_penalty=1e-8,
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
            scope="gates_pos")
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts_pos")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)

        final_probabilities_pos = tf.reshape(probabilities_by_class_and_batch,
                                         [-1, vocab_size])

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates_neg")
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts_neg")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)

        final_probabilities_neg = tf.reshape(probabilities_by_class_and_batch,
                                             [-1, vocab_size])
        final_probabilities = final_probabilities_pos/(final_probabilities_pos + final_probabilities_neg + 1e-6)

        return {"predictions": final_probabilities, "predictions_positive": final_probabilities_pos,
                "predictions_negative": final_probabilities_neg}

class MoeMaxModel(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_mixtures=None,
                     l2_penalty=1e-8,
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
            vocab_size * (num_mixtures+1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size*num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")
        expert_others = slim.fully_connected(
            model_input,
            vocab_size,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="others")

        expert_activations = tf.reshape(expert_activations,[-1,vocab_size,num_mixtures])
        forward_indices = []
        backward_indices = []
        for i in range(num_mixtures):
            forward_indice = np.arange(vocab_size)
            np.random.seed(i)
            np.random.shuffle(forward_indice)
            backward_indice = np.argsort(forward_indice,axis=None)
            forward_indices.append(forward_indice)
            backward_indices.append(backward_indice)

        forward_indices = tf.constant(np.stack(forward_indices,axis=1),dtype=tf.int32)*num_mixtures + tf.reshape(tf.range(num_mixtures),[1,-1])
        backward_indices = tf.constant(np.stack(backward_indices,axis=1),dtype=tf.int32)*num_mixtures + tf.reshape(tf.range(num_mixtures),[1,-1])
        forward_indices = tf.stop_gradient(tf.reshape(forward_indices,[-1]))
        backward_indices = tf.stop_gradient(tf.reshape(backward_indices,[-1]))

        expert_activations = tf.transpose(tf.reshape(expert_activations,[-1,vocab_size*num_mixtures]))
        expert_activations = tf.transpose(tf.gather(expert_activations,forward_indices))
        expert_activations = tf.reshape(expert_activations,[-1,vocab_size,num_mixtures])

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures+1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_softmax = tf.transpose(expert_activations,perm=[0,2,1])
        expert_softmax = tf.concat((tf.reshape(expert_softmax,[-1,num_mixtures]),tf.reshape(expert_others,[-1,1])),axis=1)
        expert_distribution = tf.nn.softmax(tf.reshape(
            expert_softmax,
            [-1, num_mixtures+1]))  # (Batch * #Labels) x num_mixtures

        expert_distribution = tf.reshape(expert_distribution[:,:num_mixtures],[-1,num_mixtures,vocab_size])
        expert_distribution = tf.reshape(tf.transpose(expert_distribution,perm=[0,2,1]),[-1,vocab_size*num_mixtures])

        expert_distribution = tf.transpose(tf.gather(tf.transpose(expert_distribution),backward_indices))
        expert_distribution = tf.reshape(expert_distribution,[-1,num_mixtures])
        probabilities_by_class_and_batch = tf.reduce_sum(gating_distribution[:, :num_mixtures] * expert_distribution, 1)

        final_probabilities = tf.reshape(probabilities_by_class_and_batch,[-1, vocab_size])
        final_probabilities_experts = tf.reshape(expert_distribution,[-1, vocab_size, num_mixtures])

        if FLAGS.moe_method=="ordered":
            seq = np.loadtxt("labels_ordered.out")
            tf_seq = tf.constant(seq,dtype=tf.int32)
            final_probabilities = tf.gather(tf.transpose(final_probabilities),tf_seq)
            final_probabilities = tf.transpose(final_probabilities)
        elif FLAGS.moe_method=="unordered":
            seq = np.loadtxt("labels_unordered.out")
            tf_seq = tf.constant(seq,dtype=tf.int32)
            final_probabilities = tf.gather(tf.transpose(final_probabilities),tf_seq)
            final_probabilities = tf.transpose(final_probabilities)

        return {"predictions": final_probabilities, "predictions_experts": final_probabilities_experts}

class MoeMaxMixModel(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_mixtures=None,
                     l2_penalty=1e-8,
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
        class_size = 25

        class_input = slim.fully_connected(
            model_input,
            model_input.get_shape().as_list()[1],
            activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="class_inputs")

        class_gate_activations = slim.fully_connected(
            class_input,
            class_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="class_gates")
        class_expert_activations = slim.fully_connected(
            class_input,
            class_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="class_experts")

        class_gating_distribution = tf.nn.softmax(tf.reshape(
            class_gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        class_expert_distribution = tf.nn.sigmoid(tf.reshape(
            class_expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        probabilities_by_class = tf.reduce_sum(
            class_gating_distribution[:, :num_mixtures] * class_expert_distribution, 1)
        probabilities_by_class = tf.reshape(probabilities_by_class,
                                            [-1, class_size])

        vocab_input = tf.concat((model_input,probabilities_by_class), axis=1)
        gate_activations = slim.fully_connected(
            vocab_input,
            vocab_size * (num_mixtures+1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
        expert_activations = slim.fully_connected(
            vocab_input,
            vocab_size*num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")
        expert_others = slim.fully_connected(
            vocab_input,
            vocab_size,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="others")

        expert_activations = tf.reshape(expert_activations,[-1,vocab_size,num_mixtures])
        forward_indices = []
        backward_indices = []
        for i in range(num_mixtures):
            forward_indice = np.arange(vocab_size)
            np.random.seed(i)
            np.random.shuffle(forward_indice)
            backward_indice = np.argsort(forward_indice,axis=None)
            forward_indices.append(forward_indice)
            backward_indices.append(backward_indice)

        forward_indices = tf.constant(np.stack(forward_indices,axis=1),dtype=tf.int32)*num_mixtures + tf.reshape(tf.range(num_mixtures),[1,-1])
        backward_indices = tf.constant(np.stack(backward_indices,axis=1),dtype=tf.int32)*num_mixtures + tf.reshape(tf.range(num_mixtures),[1,-1])
        forward_indices = tf.stop_gradient(tf.reshape(forward_indices,[-1]))
        backward_indices = tf.stop_gradient(tf.reshape(backward_indices,[-1]))

        expert_activations = tf.transpose(tf.reshape(expert_activations,[-1,vocab_size*num_mixtures]))
        expert_activations = tf.transpose(tf.gather(expert_activations,forward_indices))
        expert_activations = tf.reshape(expert_activations,[-1,vocab_size,num_mixtures])

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures+1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_softmax = tf.transpose(expert_activations,perm=[0,2,1])
        expert_softmax = tf.concat((tf.reshape(expert_softmax,[-1,num_mixtures]),tf.reshape(expert_others,[-1,1])),axis=1)
        expert_distribution = tf.nn.softmax(tf.reshape(
            expert_softmax,
            [-1, num_mixtures+1]))  # (Batch * #Labels) x num_mixtures

        expert_distribution = tf.reshape(expert_distribution[:,:num_mixtures],[-1,num_mixtures,vocab_size])
        expert_distribution = tf.reshape(tf.transpose(expert_distribution,perm=[0,2,1]),[-1,vocab_size*num_mixtures])

        expert_distribution = tf.transpose(tf.gather(tf.transpose(expert_distribution),backward_indices))
        expert_distribution = tf.reshape(expert_distribution,[-1,num_mixtures])
        probabilities_by_class_and_batch = tf.reduce_sum(gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        final_probabilities = tf.reshape(probabilities_by_class_and_batch,[-1, vocab_size])

        return {"predictions": final_probabilities, "predictions_class": probabilities_by_class}

class MoeKnowledgeModel(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_mixtures=None,
                     l2_penalty=1e-8,
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
        class_size = FLAGS.class_size
        shape = model_input.get_shape().as_list()[1]

        seq = np.loadtxt(FLAGS.class_file)
        tf_seq = tf.constant(seq,dtype=tf.float32)

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")
        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures
        probabilities_by_class = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        probabilities_by_class = tf.reshape(probabilities_by_class,
                                            [-1, vocab_size])

        probabilities_by_vocab = probabilities_by_class
        vocab_input = model_input
        for i in range(FLAGS.moe_layers):
            class_input_1 = slim.fully_connected(
                probabilities_by_vocab,
                class_size,
                activation_fn=tf.nn.elu,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="class_inputs1-%s" % i)
            class_input_2 = tf.matmul(probabilities_by_vocab,tf_seq)
            class_input_2 = slim.fully_connected(
                class_input_2,
                class_size,
                activation_fn=tf.nn.elu,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="class_inputs2-%s" % i)
            class_input_1 = tf.nn.l2_normalize(class_input_1,dim=1)*tf.sqrt(tf.cast(class_size,dtype=tf.float32)/shape)
            class_input_2 = tf.nn.l2_normalize(class_input_2,dim=1)*tf.sqrt(tf.cast(class_size,dtype=tf.float32)/shape)
            vocab_input = tf.concat((vocab_input,class_input_1,class_input_2),axis=1)
            gate_activations = slim.fully_connected(
                vocab_input,
                vocab_size * (num_mixtures + 1),
                activation_fn=None,
                biases_initializer=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="gates-%s" % i)
            expert_activations = slim.fully_connected(
                vocab_input,
                vocab_size * num_mixtures,
                activation_fn=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="experts-%s" % i)

            gating_distribution = tf.nn.softmax(tf.reshape(
                gate_activations,
                [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
            expert_distribution = tf.nn.sigmoid(tf.reshape(
                expert_activations,
                [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

            probabilities_by_vocab = tf.reduce_sum(
                gating_distribution[:, :num_mixtures] * expert_distribution, 1)
            probabilities_by_vocab = tf.reshape(probabilities_by_vocab,
                                                [-1, vocab_size])
            if i<FLAGS.moe_layers-1:
                probabilities_by_class = tf.concat((probabilities_by_class,probabilities_by_vocab),axis=1)

        final_probabilities = probabilities_by_vocab

        return {"predictions": final_probabilities, "predictions_class": probabilities_by_class}


class MoeMixModel(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_mixtures=None,
                     l2_penalty=1e-8,
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
        class_size = FLAGS.encoder_size

        class_input = slim.fully_connected(
            model_input,
            model_input.get_shape().as_list()[1],
            activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="class_inputs")

        class_gate_activations = slim.fully_connected(
            class_input,
            class_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="class_gates")
        class_expert_activations = slim.fully_connected(
            class_input,
            class_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="class_experts")

        class_gating_distribution = tf.nn.softmax(tf.reshape(
            class_gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        class_expert_distribution = tf.nn.sigmoid(tf.reshape(
            class_expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        probabilities_by_class = tf.reduce_sum(
            class_gating_distribution[:, :num_mixtures] * class_expert_distribution, 1)
        probabilities_by_class = tf.reshape(probabilities_by_class,
                                            [-1, class_size])

        vocab_input = tf.concat((model_input, probabilities_by_class), axis=1)
        gate_activations = slim.fully_connected(
            vocab_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
        expert_activations = slim.fully_connected(
            vocab_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        probabilities_by_vocab = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        probabilities_by_vocab = tf.reshape(probabilities_by_vocab,
                                            [-1, vocab_size])

        final_probabilities = probabilities_by_vocab
        return {"predictions": final_probabilities, "predictions_class": probabilities_by_class}

class MoeMixExtendModel(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_mixtures=None,
                     l2_penalty=1e-8,
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
        num_extends = FLAGS.moe_num_extend

        class_size = FLAGS.encoder_size
        model_input_stop = tf.stop_gradient(model_input)
        class_input = slim.fully_connected(
            model_input_stop,
            model_input.get_shape().as_list()[1],
            activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="class_inputs")

        class_gate_activations = slim.fully_connected(
            class_input,
            class_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="class_gates")
        class_expert_activations = slim.fully_connected(
            class_input,
            class_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="class_experts")

        class_gating_distribution = tf.nn.softmax(tf.reshape(
            class_gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        class_expert_distribution = tf.nn.sigmoid(tf.reshape(
            class_expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        probabilities_by_class = tf.reduce_sum(
            class_gating_distribution[:, :num_mixtures] * class_expert_distribution, 1)
        probabilities_by_class = tf.reshape(probabilities_by_class,
                                            [-1, class_size])

        vocab_input = tf.concat((model_input, probabilities_by_class),axis=1)
        gate_activations = slim.fully_connected(
            vocab_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
        expert_activations = slim.fully_connected(
            vocab_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        probabilities_by_vocab = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)


        final_probabilities = tf.reduce_max(tf.reshape(probabilities_by_vocab,
                                                       [-1, num_extends, vocab_size]),axis=1)
        probabilities_by_class = tf.reduce_mean(tf.reshape(probabilities_by_class,
                                                          [-1, num_extends, class_size]),axis=1)

        return {"predictions": final_probabilities, "predictions_class": probabilities_by_class}

class MoeMix2Model(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_mixtures=None,
                     l2_penalty=1e-8,
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
        class_size = FLAGS.encoder_size
        hidden_channels = FLAGS.hidden_channels
        shape = model_input.get_shape().as_list()[1]

        class_input = slim.fully_connected(
            model_input,
            shape,
            activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="class_inputs")

        class_gate_activations = slim.fully_connected(
            class_input,
            class_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="class_gates")
        class_expert_activations = slim.fully_connected(
            class_input,
            class_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="class_experts")

        class_gating_distribution = tf.nn.softmax(tf.reshape(
            class_gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        class_expert_distribution = tf.nn.sigmoid(tf.reshape(
            class_expert_activations,
            [-1,class_size, num_mixtures]))  # (Batch * #Labels) x num_mixtures
        class_expert_distribution = tf.reshape(class_expert_distribution,[-1,num_mixtures])

        probabilities_by_class = tf.reduce_sum(
            class_gating_distribution[:, :num_mixtures] * class_expert_distribution, 1)
        """
        class_expert_activations = slim.fully_connected(
            class_input,
            class_size,
            activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="class_experts")
        probabilities_by_class = slim.fully_connected(
            class_expert_activations,
            class_size,
            activation_fn=tf.nn.softmax,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="probabilities_by_class")"""

        probabilities_by_class = tf.reshape(probabilities_by_class,
                                            [-1, class_size])

        vars = np.loadtxt(FLAGS.autoencoder_dir+'autoencoder_layer%d.model' % FLAGS.encoder_layers)
        weights = tf.constant(vars[:-1,:],dtype=tf.float32)
        bias = tf.reshape(tf.constant(vars[-1,:],dtype=tf.float32),[-1])
        class_output = tf.nn.relu(tf.nn.xw_plus_b(probabilities_by_class,weights,bias))

        class_output = tf.nn.l2_normalize(class_output,dim=1)*tf.sqrt(tf.cast(class_size,dtype=tf.float32)/shape)

        vocab_input = tf.concat((model_input, class_output), axis=1)
        gate_activations = slim.fully_connected(
            vocab_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
        expert_activations = slim.fully_connected(
            vocab_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        probabilities_by_vocab = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        probabilities_by_vocab = tf.reshape(probabilities_by_vocab,
                                            [-1, vocab_size])

        final_probabilities = probabilities_by_vocab

        """
        final_probabilities = tf.reshape(probabilities_by_class,[-1,class_size*hidden_channels])
        for i in range(FLAGS.encoder_layers, FLAGS.encoder_layers*2):
            var_i = np.loadtxt(FLAGS.autoencoder_dir+'autoencoder_layer%d.model' % i)
            weight_i = tf.constant(var_i[:-1,:],dtype=tf.float32)
            bias_i = tf.reshape(tf.constant(var_i[-1,:],dtype=tf.float32),[-1])
            final_probabilities = tf.nn.xw_plus_b(final_probabilities,weight_i,bias_i)
            if i<FLAGS.encoder_layers*2-1:
                final_probabilities = tf.nn.relu(final_probabilities)
            else:
                final_probabilities = tf.nn.sigmoid(final_probabilities)"""

        return {"predictions": final_probabilities, "predictions_encoder": probabilities_by_class}

class MoeMix3Model(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_mixtures=None,
                     l2_penalty=1e-8,
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
        class_size = FLAGS.encoder_size
        hidden_channels = FLAGS.hidden_channels
        shape = model_input.get_shape().as_list()[1]

        class_input = slim.fully_connected(
            model_input,
            shape,
            activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="class_inputs")

        class_gate_activations = slim.fully_connected(
            class_input,
            class_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="class_gates")
        class_expert_activations = slim.fully_connected(
            class_input,
            class_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="class_experts")

        class_gating_distribution = tf.nn.softmax(tf.reshape(
            class_gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        class_expert_distribution = tf.reshape(class_expert_activations,[-1,num_mixtures])

        probabilities_by_class = tf.reduce_sum(
            class_gating_distribution[:, :num_mixtures] * class_expert_distribution, 1)

        probabilities_by_class = tf.reshape(probabilities_by_class,
                                            [-1, class_size])

        hidden_mean = tf.reduce_mean(probabilities_by_class,axis=1,keep_dims=True)
        hidden_std = tf.sqrt(tf.reduce_mean(tf.square(probabilities_by_class-hidden_mean),axis=1,keep_dims=True))
        probabilities_by_class = (probabilities_by_class-hidden_mean)/(hidden_std+1e-6)
        hidden_2 = tf.nn.relu(probabilities_by_class)

        vars = np.loadtxt(FLAGS.autoencoder_dir+'autoencoder_layer%d.model' % FLAGS.encoder_layers)
        weights = tf.constant(vars[:-1,:],dtype=tf.float32)
        bias = tf.reshape(tf.constant(vars[-1,:],dtype=tf.float32),[-1])
        class_output = tf.nn.relu(tf.nn.xw_plus_b(hidden_2,weights,bias))

        #class_output = probabilities_by_class

        class_output = tf.nn.l2_normalize(class_output,dim=1)*tf.sqrt(tf.cast(class_size,dtype=tf.float32)/shape)

        vocab_input = tf.concat((model_input, class_output), axis=1)
        gate_activations = slim.fully_connected(
            vocab_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
        expert_activations = slim.fully_connected(
            vocab_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        probabilities_by_vocab = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        probabilities_by_vocab = tf.reshape(probabilities_by_vocab,
                                            [-1, vocab_size])

        final_probabilities = probabilities_by_vocab


        return {"predictions": final_probabilities, "predictions_encoder": probabilities_by_class}

class MoeMix4Model(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_mixtures=None,
                     l2_penalty=1e-8,
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
        class_size = FLAGS.class_size
        shape = model_input.get_shape().as_list()[1]

        if FLAGS.moe_group:
            channels = vocab_size//class_size + 1
            vocab_input = model_input
            probabilities_by_class = []
            for i in range(channels):
                if i<channels-1:
                    sub_vocab_size = class_size
                else:
                    sub_vocab_size = vocab_size - (channels-1)*class_size
                gate_activations = slim.fully_connected(
                    vocab_input,
                    sub_vocab_size * (num_mixtures + 1),
                    activation_fn=None,
                    biases_initializer=None,
                    weights_regularizer=slim.l2_regularizer(l2_penalty),
                    scope="class_gates-%s" % i)
                expert_activations = slim.fully_connected(
                    vocab_input,
                    sub_vocab_size * num_mixtures,
                    activation_fn=None,
                    weights_regularizer=slim.l2_regularizer(l2_penalty),
                    scope="class_experts-%s" % i)

                gating_distribution = tf.nn.softmax(tf.reshape(
                    gate_activations,
                    [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
                expert_distribution = tf.nn.sigmoid(tf.reshape(
                    expert_activations,
                    [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

                probabilities_by_vocab = tf.reduce_sum(
                    gating_distribution[:, :num_mixtures] * expert_distribution, 1)
                probabilities_by_vocab = tf.reshape(probabilities_by_vocab,
                                                    [-1, sub_vocab_size])
                if i==0:
                    probabilities_by_class = probabilities_by_vocab
                else:
                    probabilities_by_class = tf.concat((probabilities_by_class, probabilities_by_vocab),axis=1)
                #probabilities_by_features = tf.stop_gradient(probabilities_by_class)
                probabilities_by_features = probabilities_by_class

                class_input_1 = slim.fully_connected(
                    probabilities_by_features,
                    class_size,
                    activation_fn=tf.nn.elu,
                    weights_regularizer=slim.l2_regularizer(l2_penalty),
                    scope="class1-%s" % i)
                class_input_2 = slim.fully_connected(
                    1-probabilities_by_features,
                    class_size,
                    activation_fn=tf.nn.elu,
                    weights_regularizer=slim.l2_regularizer(l2_penalty),
                    scope="class2-%s" % i)
                if not FLAGS.frame_features:
                    class_input_1 = tf.nn.l2_normalize(class_input_1,dim=1)*tf.sqrt(tf.cast(class_size,dtype=tf.float32)/shape)
                    class_input_2 = tf.nn.l2_normalize(class_input_2,dim=1)*tf.sqrt(tf.cast(class_size,dtype=tf.float32)/shape)
                vocab_input = tf.concat((model_input,class_input_1,class_input_2),axis=1)
                """
                class_input_1 = slim.fully_connected(
                    probabilities_by_features,
                    class_size,
                    activation_fn=tf.nn.elu,
                    weights_regularizer=slim.l2_regularizer(l2_penalty),
                    scope="class1-%s" % i)
                if not FLAGS.frame_features:
                    class_input_1 = tf.nn.l2_normalize(class_input_1,dim=1)*tf.sqrt(tf.cast(class_size,dtype=tf.float32)/shape)
                vocab_input = tf.concat((model_input,class_input_1),axis=1)"""
        else:
            gate_activations = slim.fully_connected(
                model_input,
                vocab_size * (num_mixtures + 1),
                activation_fn=None,
                biases_initializer=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="gates")
            expert_activations = slim.fully_connected(
                model_input,
                vocab_size * num_mixtures,
                activation_fn=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="experts")
            gating_distribution = tf.nn.softmax(tf.reshape(
                gate_activations,
                [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
            expert_distribution = tf.nn.sigmoid(tf.reshape(
                expert_activations,
                [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures
            probabilities_by_class = tf.reduce_sum(
                gating_distribution[:, :num_mixtures] * expert_distribution, 1)
            probabilities_by_class = tf.reshape(probabilities_by_class,
                                                [-1, vocab_size])

        probabilities_by_vocab = probabilities_by_class
        vocab_input = model_input
        for i in range(FLAGS.moe_layers):
            class_input_1 = slim.fully_connected(
                probabilities_by_vocab,
                class_size,
                activation_fn=tf.nn.elu,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="class_inputs1-%s" % i)
            class_input_2 = slim.fully_connected(
                1-probabilities_by_vocab,
                class_size,
                activation_fn=tf.nn.elu,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="class_inputs2-%s" % i)
            if not FLAGS.frame_features:
                class_input_1 = tf.nn.l2_normalize(class_input_1,dim=1)*tf.sqrt(tf.cast(class_size,dtype=tf.float32)/shape)
                class_input_2 = tf.nn.l2_normalize(class_input_2,dim=1)*tf.sqrt(tf.cast(class_size,dtype=tf.float32)/shape)
            vocab_input = tf.concat((vocab_input,class_input_1,class_input_2),axis=1)
            gate_activations = slim.fully_connected(
                vocab_input,
                vocab_size * (num_mixtures + 1),
                activation_fn=None,
                biases_initializer=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="gates-%s" % i)
            expert_activations = slim.fully_connected(
                vocab_input,
                vocab_size * num_mixtures,
                activation_fn=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="experts-%s" % i)

            gating_distribution = tf.nn.softmax(tf.reshape(
                gate_activations,
                [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
            expert_distribution = tf.nn.sigmoid(tf.reshape(
                expert_activations,
                [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

            probabilities_by_vocab = tf.reduce_sum(
                gating_distribution[:, :num_mixtures] * expert_distribution, 1)
            probabilities_by_vocab = tf.reshape(probabilities_by_vocab,
                                            [-1, vocab_size])
            if i<FLAGS.moe_layers-1:
                probabilities_by_class = tf.concat((probabilities_by_class,probabilities_by_vocab),axis=1)

        final_probabilities = probabilities_by_vocab

        return {"predictions": final_probabilities, "predictions_class": probabilities_by_class}

class MoeNoiseModel(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_mixtures=None,
                     l2_penalty=1e-8,
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
        class_size = FLAGS.class_size
        shape = model_input.get_shape().as_list()[1]

        if FLAGS.train=="train":
            noise = tf.random_normal(shape=tf.shape(model_input), mean=0.0, stddev=FLAGS.noise_std, dtype=tf.float32)
            model_input = tf.nn.l2_normalize(model_input+noise, 1)

        if FLAGS.moe_group:
            channels = vocab_size//class_size + 1
            vocab_input = model_input
            probabilities_by_class = []
            for i in range(channels):
                if i<channels-1:
                    sub_vocab_size = class_size
                else:
                    sub_vocab_size = vocab_size - (channels-1)*class_size
                gate_activations = slim.fully_connected(
                    vocab_input,
                    sub_vocab_size * (num_mixtures + 1),
                    activation_fn=None,
                    biases_initializer=None,
                    weights_regularizer=slim.l2_regularizer(l2_penalty),
                    scope="class_gates-%s" % i)
                expert_activations = slim.fully_connected(
                    vocab_input,
                    sub_vocab_size * num_mixtures,
                    activation_fn=None,
                    weights_regularizer=slim.l2_regularizer(l2_penalty),
                    scope="class_experts-%s" % i)

                gating_distribution = tf.nn.softmax(tf.reshape(
                    gate_activations,
                    [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
                expert_distribution = tf.nn.sigmoid(tf.reshape(
                    expert_activations,
                    [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

                probabilities_by_vocab = tf.reduce_sum(
                    gating_distribution[:, :num_mixtures] * expert_distribution, 1)
                probabilities_by_vocab = tf.reshape(probabilities_by_vocab,
                                                    [-1, sub_vocab_size])
                if i==0:
                    probabilities_by_class = probabilities_by_vocab
                else:
                    probabilities_by_class = tf.concat((probabilities_by_class, probabilities_by_vocab),axis=1)
                #probabilities_by_features = tf.stop_gradient(probabilities_by_class)
                probabilities_by_features = probabilities_by_class
                class_input = slim.fully_connected(
                    probabilities_by_features,
                    class_size,
                    activation_fn=tf.nn.elu,
                    weights_regularizer=slim.l2_regularizer(l2_penalty),
                    scope="class-%s" % i)
                class_input = tf.nn.l2_normalize(class_input,dim=1)*tf.sqrt(tf.cast(class_size,dtype=tf.float32)/shape)
                vocab_input = tf.concat((model_input,class_input),axis=1)
        else:
            gate_activations = slim.fully_connected(
                model_input,
                vocab_size * (num_mixtures + 1),
                activation_fn=None,
                biases_initializer=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="gates")
            expert_activations = slim.fully_connected(
                model_input,
                vocab_size * num_mixtures,
                activation_fn=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="experts")
            gating_distribution = tf.nn.softmax(tf.reshape(
                gate_activations,
                [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
            expert_distribution = tf.nn.sigmoid(tf.reshape(
                expert_activations,
                [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures
            probabilities_by_class = tf.reduce_sum(
                gating_distribution[:, :num_mixtures] * expert_distribution, 1)
            probabilities_by_class = tf.reshape(probabilities_by_class,
                                                [-1, vocab_size])

        probabilities_by_vocab = probabilities_by_class
        vocab_input = model_input
        for i in range(FLAGS.moe_layers):
            class_input = slim.fully_connected(
                probabilities_by_vocab,
                class_size,
                activation_fn=tf.nn.elu,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="class_inputs-%s" % i)
            if FLAGS.train=="train":
                noise = tf.random_normal(shape=tf.shape(class_input), mean=0.0, stddev=0.2, dtype=tf.float32)
                class_input = tf.nn.l2_normalize(class_input+noise, 1)
            class_input = tf.nn.l2_normalize(class_input,dim=1)*tf.sqrt(tf.cast(class_size,dtype=tf.float32)/shape)
            vocab_input = tf.concat((vocab_input,class_input),axis=1)
            gate_activations = slim.fully_connected(
                vocab_input,
                vocab_size * (num_mixtures + 1),
                activation_fn=None,
                biases_initializer=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="gates-%s" % i)
            expert_activations = slim.fully_connected(
                vocab_input,
                vocab_size * num_mixtures,
                activation_fn=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="experts-%s" % i)

            gating_distribution = tf.nn.softmax(tf.reshape(
                gate_activations,
                [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
            expert_distribution = tf.nn.sigmoid(tf.reshape(
                expert_activations,
                [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

            probabilities_by_vocab = tf.reduce_sum(
                gating_distribution[:, :num_mixtures] * expert_distribution, 1)
            probabilities_by_vocab = tf.reshape(probabilities_by_vocab,
                                                [-1, vocab_size])
            if i<FLAGS.moe_layers-1:
                probabilities_by_class = tf.concat((probabilities_by_class,probabilities_by_vocab),axis=1)

        final_probabilities = probabilities_by_vocab

        return {"predictions": final_probabilities, "predictions_class": probabilities_by_class}

class MoeMix5Model(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_mixtures=None,
                     l2_penalty=1e-8,
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
        class_size = FLAGS.class_size
        shape = model_input.get_shape().as_list()[1]
        feature_sizes = FLAGS.feature_sizes
        feature_sizes = [int(feature_size) for feature_size in feature_sizes.split(',')]
        feature_input = model_input[:,0:feature_sizes[0]]
        probabilities_by_class = model_input[:,feature_sizes[0]:]

        class_input = slim.fully_connected(
            probabilities_by_class,
            class_size,
            activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="class_inputs")
        class_input = tf.nn.l2_normalize(class_input,dim=1)*tf.sqrt(tf.cast(class_size,dtype=tf.float32)/shape)
        vocab_input = tf.concat((feature_input,class_input),axis=1)
        gate_activations = slim.fully_connected(
            vocab_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
        expert_activations = slim.fully_connected(
            vocab_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        probabilities_by_vocab = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        probabilities_by_vocab = tf.reshape(probabilities_by_vocab,
                                            [-1, vocab_size])
        final_probabilities = probabilities_by_vocab

        return {"predictions": final_probabilities}

class MoeExtendModel(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_mixtures=None,
                     l2_penalty=1e-8,
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
        num_extends = FLAGS.moe_num_extend

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures


        final_probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)

        final_probabilities = tf.reduce_max(tf.reshape(final_probabilities_by_class_and_batch,
                                       [-1, num_extends, vocab_size]), axis=1)

        return {"predictions": final_probabilities}

class MoeExtendDistillChainModel(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     distill_labels=None,
                     num_mixtures=None,
                     l2_penalty=1e-8,
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
        num_extends = FLAGS.moe_num_extend
        class_size = 256
        if distill_labels is not None:
            class_input = slim.fully_connected(
                distill_labels,
                class_size,
                activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="class_inputs")
            class_input = tf.reshape(tf.tile(tf.reshape(class_input,[-1,1,class_size]),[1,num_extends,1]),[-1,class_size])
            model_input = tf.concat((model_input,class_input),axis=1)
        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures


        final_probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)

        final_probabilities = tf.reduce_max(tf.reshape(final_probabilities_by_class_and_batch,
                                                       [-1, num_extends, vocab_size]), axis=1)

        return {"predictions": final_probabilities}

class MoeExtendCombineModel(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_mixtures=None,
                     l2_penalty=1e-8,
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
        class_size = FLAGS.class_size
        num_extends = FLAGS.moe_num_extend
        shape = model_input.get_shape().as_list()[1]
        model_input = tf.reshape(model_input,[-1, num_extends, shape])
        model_input_0 = model_input[:,0,:]
        gate_activations = slim.fully_connected(
            model_input_0,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
        expert_activations = slim.fully_connected(
            model_input_0,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")
        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures
        probabilities_by_class = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        probabilities_by_class = tf.reshape(probabilities_by_class,
                                            [-1, vocab_size])

        probabilities_by_vocab = probabilities_by_class
        input_layers = []
        for i in range(FLAGS.moe_layers-1):
            model_input_i = model_input[:,i+1,:]
            class_input_1 = slim.fully_connected(
                probabilities_by_vocab,
                class_size,
                activation_fn=tf.nn.elu,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="class_inputs1-%s" % i)
            class_input_1 = tf.nn.l2_normalize(class_input_1,dim=1)*tf.sqrt(tf.cast(class_size,dtype=tf.float32)/shape)
            input_layers.append(class_input_1)
            vocab_input = tf.concat([model_input_i]+input_layers,axis=1)
            gate_activations = slim.fully_connected(
                vocab_input,
                vocab_size * (num_mixtures + 1),
                activation_fn=None,
                biases_initializer=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="gates-%s" % i)
            expert_activations = slim.fully_connected(
                vocab_input,
                vocab_size * num_mixtures,
                activation_fn=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="experts-%s" % i)

            gating_distribution = tf.nn.softmax(tf.reshape(
                gate_activations,
                [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
            expert_distribution = tf.nn.sigmoid(tf.reshape(
                expert_activations,
                [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

            probabilities_by_vocab = tf.reduce_sum(
                gating_distribution[:, :num_mixtures] * expert_distribution, 1)
            probabilities_by_vocab = tf.reshape(probabilities_by_vocab,
                                                [-1, vocab_size])

            probabilities_by_class = tf.concat((probabilities_by_class,probabilities_by_vocab),axis=1)

        final_probabilities = probabilities_by_vocab

        return {"predictions": final_probabilities, "predictions_class": probabilities_by_class}

class MoeExtendSoftmaxModel(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_mixtures=None,
                     l2_penalty=1e-8,
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
        num_extends = FLAGS.moe_num_extend

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        extend_activations = slim.fully_connected(
            model_input,
            vocab_size,
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="extends")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures
        extend_distribution = tf.nn.softmax(tf.reshape(
            extend_activations,
            [-1, num_extends, vocab_size]),dim=1)  # (Batch * #Labels) x (num_mixtures + 1)

        final_probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)

        final_probabilities = tf.reduce_sum(tf.reshape(final_probabilities_by_class_and_batch,
                                        [-1, num_extends, vocab_size])*extend_distribution,axis=1)
        return {"predictions": final_probabilities}

class MoeSepModel(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_mixtures=None,
                     l2_penalty=1e-8,
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
        feature_sizes = FLAGS.feature_sizes
        feature_sizes = [int(feature_size) for feature_size in feature_sizes.split(',')]
        fbegin = 0
        final_probabilities_all = []
        for i in range(len(feature_sizes)):
            feature_size = feature_sizes[i]
            feature_input = model_input[:,fbegin:fbegin+feature_size]
            fbegin += feature_size
            gate = slim.fully_connected(
                feature_input,
                vocab_size * (num_mixtures + 1),
                activation_fn=None,
                biases_initializer=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="gates-%s" % i)
            expert = slim.fully_connected(
                feature_input,
                vocab_size * num_mixtures,
                activation_fn=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="experts-%s" % i)
            gating_distribution = tf.nn.softmax(tf.reshape(
                gate,
                [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
            expert_distribution = tf.nn.sigmoid(tf.reshape(
                expert,
                [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

            final_prob = tf.reduce_sum(
                gating_distribution[:, :num_mixtures] * expert_distribution, 1)
            final_prob = tf.reshape(final_prob,[-1, vocab_size])

            final_probabilities_all.append(final_prob)

        final_probabilities_all = tf.stack(final_probabilities_all,axis=1)
        final_probabilities = tf.reduce_max(final_probabilities_all,axis=1)
        return {"predictions": final_probabilities}

class SimModel(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_mixtures=None,
                     l2_penalty=1e-8,
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
        embedding_size = model_input.get_shape().as_list()[1]

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)

        model_input = tf.maximum(model_input,tf.zeros_like(model_input))
        expert_distribution = []
        for i in range(num_mixtures):
            embeddings = tf.Variable(tf.truncated_normal([vocab_size, embedding_size],stddev=0.1))
            tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(embeddings))
            embeddings = tf.maximum(embeddings,tf.zeros_like(embeddings))
            norm_embeddings = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = tf.div(embeddings, norm_embeddings)
            norm_input = tf.sqrt(tf.reduce_sum(tf.square(model_input), 1, keep_dims=True))
            normalized_input = tf.div(model_input,norm_input)
            similarity = tf.matmul(normalized_input, normalized_embeddings, transpose_b=True)*2
            expert_distribution.append(similarity)

        expert_distribution = tf.stack(expert_distribution,axis=2)
        expert_distribution = tf.reshape(expert_distribution,[-1,num_mixtures])

        probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)

        probabilities_by_class_and_batch = tf.reshape(probabilities_by_class_and_batch,
                                                      [-1, vocab_size])

        final_probabilities = tf.reshape(probabilities_by_class_and_batch,
                                         [-1, vocab_size])

        return {"predictions": final_probabilities}

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

        model_input = model_input

        hidden_size_1 = FLAGS.hidden_size_1
        hidden_size_2 = FLAGS.encoder_size
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
                activation_fn=tf.nn.relu,
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
        """
        scale = tf.get_variable("scale", [1, vocab_size], tf.float32,
                                   initializer=tf.constant_initializer(0.0))
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(scale))"""

        output_2 = model_input
        return {"predictions": output_2}