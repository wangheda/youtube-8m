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

class LstmAttentionLstmModel(models.BaseModel):

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
    l2_penalty = unused_params.get("l2_penalty", 1e-8)
    max_frames = model_input.get_shape().as_list()[1]

    mask_array = []
    for i in xrange(max_frames + 1):
      tmp = [0.0] * max_frames 
      for j in xrange(i):
        tmp[j] = 1.0
      mask_array.append(tmp)
    mask_array = np.array(mask_array)
    mask_init = tf.constant_initializer(mask_array)
    mask_emb = tf.get_variable("mask_emb", shape = [max_frames + 1, max_frames], 
            dtype = tf.float32, trainable = False, initializer = mask_init)
    
    ## Batch normalize the input
    stacked_cell = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=True)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=True)

    stacked_att_cell = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=True)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=True)

    loss = 0.0
    with tf.variable_scope("ATT"):
      att_outputs, att_state = tf.nn.dynamic_rnn(stacked_att_cell, model_input,
                                         sequence_length=num_frames, 
                                         swap_memory=FLAGS.rnn_swap_memory,
                                         dtype=tf.float32)
      att_final_memory = tf.concat(map(lambda x: x.c, att_state), axis = 1)

    with tf.variable_scope("RNN"):
      outputs, state = tf.nn.dynamic_rnn(stacked_cell, model_input,
                                         sequence_length=num_frames, 
                                         swap_memory=FLAGS.rnn_swap_memory,
                                         dtype=tf.float32)
      final_memory = tf.concat(map(lambda x: x.c, state), axis = 1)

      num_frames_matrix = tf.maximum(tf.cast(
          tf.expand_dims(tf.expand_dims(num_frames, axis=1), axis=2), 
          dtype=tf.float32), 1.0)
      mask = tf.expand_dims(tf.nn.embedding_lookup(mask_emb, num_frames), axis = 2)
      attention_fc = slim.fully_connected(
          att_outputs, 1, activation_fn=tf.nn.sigmoid,
          weights_regularizer=slim.l2_regularizer(l2_penalty))
      print attention_fc
      
      attention = attention_fc * mask
      attention_sum = tf.reduce_sum(attention, axis = 1, keep_dims = True) + 1e-8
      attention = attention / attention_sum
      print attention

      attended_output = tf.reduce_sum(attention * outputs, axis = 1)
      final_state = tf.concat([attended_output, att_final_memory, final_memory], axis = 1)
      print "final_state", final_state

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=final_state,
        original_input=model_input,
        vocab_size=vocab_size,
        **unused_params)

