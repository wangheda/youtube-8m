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

class CnnLstmMemoryMultiTaskModel(models.BaseModel):

  def cnn(self, 
          model_input, 
          l2_penalty=1e-8, 
          num_filters = [1024, 1024, 1024],
          filter_sizes = [1,2,3], 
          **unused_params):
    max_frames = model_input.get_shape().as_list()[1]
    num_features = model_input.get_shape().as_list()[2]

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
    max_frames = model_input.get_shape().as_list()[1]

    cnn_output = self.cnn(model_input, num_filters=[1024,1024,1024], filter_sizes=[1,2,3])
    normalized_cnn_output = tf.nn.l2_normalize(cnn_output, dim=2)
    
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
      outputs, state = tf.nn.dynamic_rnn(stacked_lstm, normalized_cnn_output,
                                         sequence_length=num_frames, 
                                         swap_memory=FLAGS.rnn_swap_memory,
                                         dtype=tf.float32)
      final_state = tf.concat(map(lambda x: x.c, state), axis = 1)

    mask = self.get_mask(max_frames, num_frames)
    mean_cnn_output = tf.einsum("ijk,ij->ik", normalized_cnn_output, mask) \
                      / tf.expand_dims(tf.cast(num_frames, dtype=tf.float32), dim=1)
    support_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_support_model)
    support_predictions = support_model().create_model(
        model_input=mean_cnn_output,
        original_input=model_input,
        vocab_size=vocab_size,
        num_mixtures=2,
        sub_scope="support",
        **unused_params)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    predictions = aggregated_model().create_model(
        model_input=final_state,
        original_input=model_input,
        vocab_size=vocab_size,
        sub_scope="main",
        **unused_params)
    return {"predictions": predictions["predictions"], 
            "support_predictions": support_predictions["predictions"]}

  def get_mask(self, max_frames, num_frames):
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
    mask = tf.nn.embedding_lookup(mask_emb, num_frames)
    return mask
