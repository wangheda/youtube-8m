import math

import models
import tensorflow as tf
import numpy as np
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS


class MoeCombineLayersModel(models.BaseModel):
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
                    scope="class-%s" % i)
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