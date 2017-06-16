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

class LstmMultiPoolingModel(models.BaseModel):

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
    batch_size = FLAGS.batch_size

    pooled_output = []
    outputs = model_input

    num_frames_matrix = tf.maximum(tf.cast(tf.expand_dims(num_frames, axis=1), dtype=tf.float32), tf.ones([batch_size, 1]))
    pooled_output.append(tf.reduce_sum(outputs, axis = 1) / num_frames_matrix)

    for layer in xrange(number_of_layers):
      ## Batch normalize the input
      cell = tf.contrib.rnn.BasicLSTMCell(
                  lstm_size, forget_bias=1.0, state_is_tuple=False)

      loss = 0.0
      with tf.variable_scope("RNN-layer%d" % layer):
        outputs, state = tf.nn.dynamic_rnn(cell, outputs,
                                           sequence_length=num_frames,
                                           time_major=False,
                                           swap_memory=FLAGS.rnn_swap_memory,
                                           dtype=tf.float32)
        pooled_output.append(tf.reduce_sum(outputs, axis = 1) / num_frames_matrix)
    final_output = tf.concat(pooled_output, axis = 1)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=final_output,
        original_input=model_input,
        vocab_size=vocab_size,
        **unused_params)

