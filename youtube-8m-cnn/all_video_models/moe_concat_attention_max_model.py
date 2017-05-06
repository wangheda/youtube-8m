import math

import models
import tensorflow as tf
import numpy as np
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS


class MoeConcatAttentionMaxModel(models.BaseModel):
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