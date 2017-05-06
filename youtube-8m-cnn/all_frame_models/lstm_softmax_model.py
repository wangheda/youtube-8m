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

class LstmSoftmaxModel(models.BaseModel):

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
        lstm_size = FLAGS.lstm_cells
        number_of_layers = FLAGS.lstm_layers
        shape = model_input.get_shape().as_list()
        num_mixtures = FLAGS.moe_num_mixtures

        frames_sum = tf.reduce_sum(tf.abs(model_input),axis=2)
        frames_true = tf.ones(tf.shape(frames_sum))
        frames_false = tf.zeros(tf.shape(frames_sum))
        frames_bool = tf.reshape(tf.where(tf.greater(frames_sum, frames_false), frames_true, frames_false),[-1,shape[1],1])

        ## Batch normalize the input
        """
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)

        with tf.variable_scope("RNN"):
            outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                               sequence_length=num_frames,
                                               swap_memory=True,
                                               dtype=tf.float32)"""
        lstm_size = shape[2]
        outputs = model_input
        softmax_input = tf.reduce_mean(tf.reshape(outputs,[-1,FLAGS.stride_size,lstm_size]),axis=1)
        frames_bool = frames_bool[:,0:shape[1]:FLAGS.stride_size,:]
        expert_activations = slim.fully_connected(
            softmax_input,
            num_mixtures*vocab_size,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="expert_activations")
        gate_activations = slim.fully_connected(
            softmax_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gate_activations")
        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        probabilities_by_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)


        gate = slim.fully_connected(
            softmax_input, vocab_size, activation_fn=None,
            weights_regularizer=slim.l2_regularizer(1e-8),
            scope="gates")

        gate = tf.nn.softmax(tf.reshape(gate,[-1, shape[1]//FLAGS.stride_size,vocab_size]),dim=1)
        gate = gate*frames_bool
        gate = gate/tf.reduce_sum(gate, axis=1, keep_dims=True)

        output = tf.reshape(probabilities_by_batch,[-1, shape[1]//FLAGS.stride_size,vocab_size])
        perdiction_frames = output
        final_probabilities = tf.reduce_sum(perdiction_frames*gate,axis=1)
        return {"predictions": final_probabilities}