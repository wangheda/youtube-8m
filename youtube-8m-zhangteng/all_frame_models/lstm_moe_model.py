import math

import models
import video_level_models
import tensorflow as tf
import model_utils as utils
import numpy as np

import tensorflow.contrib.slim as slim
from tensorflow import flags
import rnn_residual

FLAGS = flags.FLAGS

class LstmMoeModel(models.BaseModel):

    def create_model(self, model_input, vocab_size, num_frames, l2_penalty=1e-8, **unused_params):
        """Creates a model which uses a stack of LSTMs to represent the video.

        Args:
          model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                       input features.
          vocab_size: The number of classes in the dataset.
          num_frames: A vector of length 'batch' which indicates the number of
               frames for each video (before padding).

        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          'batch_size' x 'num_classes'.
        """
        shape = model_input.get_shape().as_list()
        lstm_size = FLAGS.lstm_cells
        number_of_layers = FLAGS.lstm_layers
        num_mixtures = 2

        ## Batch normalize the input
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(
            lstm_size, forget_bias=1.0, state_is_tuple=False)

        with tf.variable_scope("RNN"):
            states = []
            layer_input = model_input
            for i in range(number_of_layers):
                outputs, state = tf.nn.dynamic_rnn(lstm_cell, layer_input,
                                                   sequence_length=num_frames,
                                                   swap_memory=True,
                                                   dtype=tf.float32,scope="lstm-%s" % i)
                states.append(state)
                if i<number_of_layers-1:
                    outputs = tf.reshape(outputs,[-1,lstm_size])
                    gate_activations = slim.fully_connected(
                        outputs,
                        lstm_size * (num_mixtures + 1),
                        activation_fn=None,
                        biases_initializer=None,
                        weights_regularizer=slim.l2_regularizer(l2_penalty),
                        scope="gates-%s" % i)
                    expert_activations = slim.fully_connected(
                        outputs,
                        lstm_size * num_mixtures,
                        activation_fn=None,
                        weights_regularizer=slim.l2_regularizer(l2_penalty),
                        scope="experts-%s" % i)
                    gating_distribution = tf.nn.softmax(tf.reshape(
                        gate_activations,
                        [-1, num_mixtures + 1]))
                    expert_distribution = tf.reshape(
                        expert_activations,
                        [-1, num_mixtures])
                    final_probabilities = tf.reduce_sum(
                        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
                    layer_input = tf.reshape(final_probabilities,[-1,shape[1],lstm_size])

        state = tf.concat(states,axis=1)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=state,
            vocab_size=vocab_size,
            **unused_params)