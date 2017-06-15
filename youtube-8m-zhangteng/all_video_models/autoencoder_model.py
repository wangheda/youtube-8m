import math

import models
import tensorflow as tf
import numpy as np
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim
FLAGS = flags.FLAGS


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