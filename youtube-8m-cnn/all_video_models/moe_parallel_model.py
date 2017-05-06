import math

import models
import tensorflow as tf
import numpy as np
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS


class MoeParallelModel(models.BaseModel):
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