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

class LstmShortLayerModel(models.BaseModel):

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
        unit_layer_1 = FLAGS.lstm_length
        unit_layer_2 = shape[1]//FLAGS.lstm_length
        model_input = model_input[:,:unit_layer_1*unit_layer_2,:]

        model_input_1 = tf.reshape(model_input,[-1,unit_layer_1,shape[2]])
        ## Batch normalize the input
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=True)
                for _ in range(1)
                ],
            state_is_tuple=True)

        with tf.variable_scope("RNN-1"):
            outputs_1, state_1 = tf.nn.dynamic_rnn(stacked_lstm, model_input_1,
                                                   sequence_length=None,
                                                   swap_memory=True,
                                                   dtype=tf.float32)
        state_c = tf.concat(map(lambda x: x.c, state_1), axis=1)
        model_input_2 = tf.reshape(state_c,[-1,unit_layer_2,lstm_size])
        num_frames = num_frames//unit_layer_1
        with tf.variable_scope("RNN-2"):
            outputs_2, state_2 = tf.nn.dynamic_rnn(stacked_lstm, model_input_2,
                                                   sequence_length=num_frames,
                                                   swap_memory=True,
                                                   dtype=tf.float32)

        state_out = tf.concat(map(lambda x: x.c, state_2), axis=1)
        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=state_out,
            vocab_size=vocab_size,
            **unused_params)