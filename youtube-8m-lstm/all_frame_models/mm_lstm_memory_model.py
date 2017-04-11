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

class MatchingMatrixLstmMemoryModel(models.BaseModel):

  def matching_matrix(self, 
          model_input, 
          vocab_size,
          l2_penalty=1e-8, 
          **unused_params):
    max_frames = model_input.get_shape().as_list()[1]
    num_features = model_input.get_shape().as_list()[2]
    embedding_size = FLAGS.mm_label_embedding

    model_input = tf.reshape(model_input, [-1, num_features])

    frame_relu = slim.fully_connected(
        model_input,
        embedding_size,
        activation_fn=tf.nn.relu,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="mm_relu")

    frame_activation = slim.fully_connected(
        frame_relu,
        embedding_size,
        activation_fn=tf.nn.tanh,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="mm_activation")

    label_embedding = tf.get_variable("label_embedding", shape=[vocab_size,embedding_size],
        dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.5),
        regularizer=slim.l2_regularizer(l2_penalty), trainable=True)

    mm_matrix = tf.einsum("ik,jk->ij", frame_activation, label_embedding)
    mm_output = tf.reshape(mm_matrix, [-1,max_frames,vocab_size])
    return mm_output

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

    mm_output = self.matching_matrix(model_input, vocab_size)
    normalized_mm_output = tf.nn.l2_normalize(mm_output, dim=2)
    
    ## Batch normalize the input
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=True)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=True)

    loss = 0.0
    with tf.variable_scope("RNN"):
      outputs, state = tf.nn.dynamic_rnn(stacked_lstm, normalized_mm_output,
                                         sequence_length=num_frames, 
                                         swap_memory=FLAGS.rnn_swap_memory,
                                         dtype=tf.float32)
      final_state = tf.concat(map(lambda x: x.c, state), axis = 1)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=final_state,
        original_input=model_input,
        vocab_size=vocab_size,
        **unused_params)

