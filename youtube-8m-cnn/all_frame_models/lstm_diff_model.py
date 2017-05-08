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

class LstmDiffModel(models.BaseModel):

    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
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

        frames_sum = tf.reduce_sum(tf.abs(model_input),axis=2)
        frames_true = tf.ones(tf.shape(frames_sum))
        frames_false = tf.zeros(tf.shape(frames_sum))
        frames_bool = tf.reshape(tf.where(tf.greater(frames_sum, frames_false), frames_true, frames_false),[-1,shape[1],1])

        model_mean = tf.reduce_sum(model_input*frames_bool,axis=1,keep_dims=True)/tf.reduce_sum(frames_bool,axis=1,keep_dims=True)
        model_input = (model_input - model_mean)*frames_bool
        model_input = tf.nn.l2_normalize(model_input, 2)

        ## Batch normalize the input
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=True)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=True)

        with tf.variable_scope("RNN"):
            outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                               sequence_length=num_frames,
                                               swap_memory=True,
                                               dtype=tf.float32)
        state_c = tf.concat(map(lambda x: x.c, state), axis=1)
        model_mean = tf.nn.l2_normalize(tf.reshape(model_mean,[-1,shape[2]]), dim=1)
        rnn_out = tf.nn.l2_normalize(state_c, dim=1)
        model_output = tf.concat((model_mean,rnn_out),axis=1)
        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=model_output,
            vocab_size=vocab_size,
            **unused_params)