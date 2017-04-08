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

class CnnLstmMemoryNormalizationModel(models.BaseModel):

  def cnn(self, 
          model_input, 
          l2_penalty=1e-8, 
          num_filters = [1024, 1024, 1024],
          filter_sizes = [1,2,3], 
          **unused_params):
    max_frames = model_input.get_shape().as_list()[1]
    num_features = model_input.get_shape().as_list()[2]
    normalize_class = getattr(self, FLAGS.lstm_normalization, self.identical)

    shift_inputs = []
    for i in xrange(max(filter_sizes)):
      if i == 0:
        shift_inputs.append(model_input)
      else:
        shift_inputs.append(tf.pad(model_input, paddings=[[0,0],[i,0],[0,0]])[:,:max_frames,:])

    cnn_outputs = []
    for nf, fs in zip(num_filters, filter_sizes):
      sub_input = tf.concat(shift_inputs[:fs], axis=2)
      sub_filter = tf.get_variable("cnn-filter-len%d"%fs, shape=[num_features*fs, nf], dtype=tf.float32, 
                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1), 
                       regularizer=tf.contrib.layers.l2_regularizer(l2_penalty))
      cnn_outputs.append(tf.einsum("ijk,kl->ijl", sub_input, sub_filter))

    cnn_output = tf.concat(cnn_outputs, axis=2)
    return cnn_output

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

    cnn_output = self.cnn(model_input, num_filters=[1024,1024,1024], filter_sizes=[1,2,3])
    
    outputs, state = cnn_output, []
    for layer in xrange(number_of_layers):
      with tf.variable_scope("RNN-layer%d" % layer):
        outputs = normalize_class(outputs)
        cell = tf.contrib.rnn.BasicLSTMCell(
                        lstm_size, forget_bias=1.0, state_is_tuple=True)
        layer_outputs, layer_state = tf.nn.dynamic_rnn(cell, outputs,
                                       sequence_length=num_frames, 
                                       swap_memory=FLAGS.rnn_swap_memory,
                                       dtype=tf.float32)
        state.append(layer_state)
        outputs = layer_outputs
      final_state = tf.concat(map(lambda x: x.c, state), axis = 1)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=final_state,
        original_input=model_input,
        vocab_size=vocab_size,
        **unused_params)

  def layer_normalize(self, input_raw, epsilon=1e-8):
    feature_dim = len(input_raw.get_shape()) - 1
    mean_input = tf.reduce_mean(input_raw, axis=feature_dim, keep_dims=True)
    std_input = tf.sqrt(tf.reduce_mean(tf.square(input_raw-mean_input), axis=feature_dim, keep_dims=True))
    std_input = tf.maximum(std_input, epsilon)
    output = (input_raw - mean_input) / std_input
    return output

  def l2_normalize(self, input_raw, epsilon=1e-8):
    feature_dim = len(input_raw.get_shape()) - 1
    output = tf.nn.l2_normalize(input_raw, dim=feature_dim)
    return output

  def identical(self, input_raw, epsilon=1e-8):
    return input_raw
