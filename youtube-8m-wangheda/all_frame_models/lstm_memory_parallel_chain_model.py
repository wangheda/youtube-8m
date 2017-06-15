import sys
import models
import model_utils
import math
import numpy as np
import video_level_models
import tensorflow as tf
import utils
import tensorflow.contrib.slim as slim
from tensorflow import flags
FLAGS = flags.FLAGS

class LstmMemoryParallelChainModel(models.BaseModel):
  """Classifier chain model of lstm memory"""

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
    lstm_size = int(FLAGS.lstm_cells)
    support_lstm_size = lstm_size / 2
    number_of_layers = FLAGS.lstm_layers
    num_supports = FLAGS.num_supports
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    stacked_lstm_support = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    support_lstm_size, forget_bias=1.0, state_is_tuple=True)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=True)

    with tf.variable_scope("RNN-support"):
      support_outputs, support_state = tf.nn.dynamic_rnn(stacked_lstm_support, model_input,
                                         sequence_length=num_frames, 
                                         swap_memory=FLAGS.rnn_swap_memory,
                                         dtype=tf.float32)
      support_final_state = tf.concat(map(lambda x: x.c, support_state), axis = 1)

    support_predictions = aggregated_model().create_model(
        model_input=support_final_state,
        original_input=model_input,
        vocab_size=num_supports,
        sub_scope="support",
        **unused_params)
    support_predictions = support_predictions["predictions"]

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=True)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=True)

    with tf.variable_scope("RNN-main"):
      outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                         sequence_length=num_frames, 
                                         swap_memory=FLAGS.rnn_swap_memory,
                                         dtype=tf.float32)
      final_state = tf.concat(map(lambda x: x.c, state), axis = 1)

    main_state = tf.concat([final_state, support_predictions], axis=1)
    predictions = aggregated_model().create_model(
        model_input=main_state,
        original_input=model_input,
        vocab_size=vocab_size,
        sub_scope="main",
        **unused_params)
    predictions["support_predictions"] = support_predictions
    return predictions

