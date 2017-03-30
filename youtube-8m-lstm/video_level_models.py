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
    "moe_num_mixtures", 2,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")
flags.DEFINE_integer(
    "moe_num_extend", 8,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")
flags.DEFINE_string("moe_method", "none",
                    "The pooling method used in the DBoF cluster layer. "
                    "Choices are 'average' and 'max'.")

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

class Stage2LogisticModel(models.BaseModel):
  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    output = tf.nn.sigmoid(model_input + slim.fully_connected(
        model_input, vocab_size, activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty)))
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
    dims = tf.rank(model_input)
    shape = model_input.get_shape().as_list()
    model_input = tf.cond(tf.equal(dims,tf.constant(3)), lambda: tf.reshape(model_input,[-1,shape[-1]]), lambda: model_input)
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
    final_probabilities = tf.cond(tf.equal(dims,tf.constant(3)), lambda: tf.reduce_mean(tf.reshape(final_probabilities,[-1,shape[1],
                                vocab_size]),axis=1), lambda: final_probabilities)
    return {"predictions": final_probabilities}

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
            vocab_size,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")
        expert_others = slim.fully_connected(
            model_input,
            vocab_size,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="others")

        expert_activations = tf.tile(tf.reshape(expert_activations,[-1,vocab_size,1]),[1,1,num_mixtures])

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures+1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_softmax = tf.transpose(expert_activations,perm=[0,2,1])
        expert_softmax = tf.concat((tf.reshape(expert_softmax,[-1,num_mixtures]),tf.reshape(expert_others,[-1,1])),axis=1)
        expert_distribution = tf.nn.softmax(tf.reshape(
            expert_softmax,
            [-1, num_mixtures+1]))  # (Batch * #Labels) x num_mixtures
        expert_distribution = tf.reshape(expert_distribution[:,:num_mixtures],[-1,vocab_size,num_mixtures])
        expert_distribution = tf.reshape(tf.transpose(expert_distribution,perm=[0,2,1]),[-1,num_mixtures])
        probabilities_by_class_and_batch = tf.reduce_sum(gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        probabilities_by_class_and_batch = tf.reshape(probabilities_by_class_and_batch,
                                                      [-1, vocab_size])

        final_probabilities = tf.reshape(probabilities_by_class_and_batch,[-1, vocab_size])

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

        return {"predictions": final_probabilities}

class MoeLevelModel(models.BaseModel):
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

        probabilities_by_vocab = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        probabilities_by_vocab = tf.reshape(probabilities_by_vocab,
                                                      [-1, vocab_size])


        class_gate_activations = slim.fully_connected(
            model_input,
            class_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="class_gates")
        class_expert_activations = slim.fully_connected(
            model_input,
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
        seq = np.loadtxt("labels_class.out")
        tf_seq = tf.constant(seq,dtype=tf.float32)
        probabilities_by_class = tf.matmul(probabilities_by_class,tf_seq)


        final_probabilities = probabilities_by_vocab*probabilities_by_class
        return {"predictions": final_probabilities}


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
        class_size = 25

        class_input = slim.fully_connected(
            model_input,
            model_input.get_shape().as_list()[1],
            activation_fn=tf.nn.tanh,
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

        vocab_input = tf.concat((model_input,class_input),axis=1)
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
            [-1, num_extends,vocab_size]), dim=1)  # (Batch * #Labels) x (num_mixtures + 1)

        final_probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)

        #final_probabilities = tf.reduce_max(tf.reshape(final_probabilities_by_class_and_batch,
        #                                [-1, num_extends, vocab_size]),axis=1)

        final_probabilities = tf.reduce_sum(tf.reshape(final_probabilities_by_class_and_batch,
                                       [-1, num_extends, vocab_size])*extend_distribution,axis=1)
        return {"predictions": final_probabilities}

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