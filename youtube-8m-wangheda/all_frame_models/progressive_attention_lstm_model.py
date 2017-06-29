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

class ProgressiveAttentionLstmModel(models.BaseModel):

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
    number_of_layers = FLAGS.lstm_layers
    num_attentions = FLAGS.num_attentions 
    print model_input.get_shape().as_list()
    max_frames = model_input.get_shape().as_list()[1]

    ## Batch normalize the input
    stacked_cell = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=True)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=True)

    att_cell = tf.contrib.rnn.AttentionCellWrapper(cell = tf.contrib.rnn.BasicLSTMCell(
            lstm_size, forget_bias=1.0, state_is_tuple=True),
            attn_length=1, state_is_tuple=True)

    loss = 0.0
    with tf.variable_scope("RNN"):
      outputs, state = tf.nn.dynamic_rnn(stacked_cell, model_input,
                                         sequence_length=num_frames, 
                                         swap_memory=FLAGS.rnn_swap_memory,
                                         dtype=tf.float32)
      final_memory = tf.concat(map(lambda x: x.c, state), axis = 1)

    with tf.variable_scope("ATT"):
      att_outputs, att_state = tf.nn.dynamic_rnn(att_cell, outputs,
                                         sequence_length=tf.ones_like(num_frames, dtype=tf.int32)*num_attentions, 
                                         swap_memory=FLAGS.rnn_swap_memory,
                                         dtype=tf.float32)
      print att_outputs
      print att_state
      att_state, _, _ = att_state
      print att_state
      att_final_memory = att_state.c

    final_state = tf.concat([att_final_memory, final_memory], axis = 1)
    print "final_state", final_state

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    predictions = aggregated_model().create_model(
        model_input=final_state,
        original_input=model_input,
        vocab_size=vocab_size,
        **unused_params)
    print predictions
    return predictions


