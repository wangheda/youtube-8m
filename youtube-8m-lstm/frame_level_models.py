# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains a collection of models which operate on variable-length sequences.
"""
import math

import models
import video_level_models
import tensorflow as tf
import model_utils
import utils

import tensorflow.contrib.slim as slim
from tensorflow import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("iterations", 30,
                     "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")
flags.DEFINE_integer("dbof_cluster_size", 8192,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 1024,
                     "Number of units in the DBoF hidden layer.")
flags.DEFINE_string("dbof_pooling_method", "max",
                    "The pooling method used in the DBoF cluster layer. "
                    "Choices are 'average' and 'max'.")
flags.DEFINE_string("video_level_classifier_model", "MoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")
flags.DEFINE_bool("rnn_swap_memory", False, "If true, swap_memory = True.")
flags.DEFINE_string("lstm_cells", "1024", "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")
flags.DEFINE_integer("gru_cells", 1024, "Number of GRU cells.")
flags.DEFINE_integer("gru_layers", 2, "Number of GRU layers.")

flags.DEFINE_string("cnn_filter_sizes", "1,2,3", "Sizes of cnn filters.")
flags.DEFINE_string("cnn_filter_nums", "256,256,256", "Numbers of every cnn filters.")
flags.DEFINE_integer("cnn_pooling_k", 4, "The k value for max-k pooling.")

class FrameLevelLogisticModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a logistic classifier over the average of the
    frame-level features.

    This class is intended to be an example for implementors of frame level
    models. If you want to train a model over averaged features it is more
    efficient to average them beforehand rather than on the fly.

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
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    feature_size = model_input.get_shape().as_list()[2]

    denominators = tf.reshape(
        tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
    avg_pooled = tf.reduce_sum(model_input,
                               axis=[1]) / denominators

    output = slim.fully_connected(
        avg_pooled, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(1e-8))
    return {"predictions": output}

class DbofModel(models.BaseModel):
  """Creates a Deep Bag of Frames model.

  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.

  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.

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

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = model_utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = model_utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    cluster_weights = tf.Variable(tf.random_normal(
        [feature_size, cluster_size],
        stddev=1 / math.sqrt(feature_size)))
    tf.summary.histogram("cluster_weights", cluster_weights)
    activation = tf.matmul(reshaped_input, cluster_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="cluster_bn")
    else:
      cluster_biases = tf.Variable(
          tf.random_normal(
              [cluster_size], stddev=1 / math.sqrt(feature_size)))
      tf.summary.histogram("cluster_biases", cluster_biases)
      activation += cluster_biases
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("cluster_output", activation)

    activation = tf.reshape(activation, [-1, max_frames, cluster_size])
    activation = model_utils.FramePooling(activation, FLAGS.dbof_pooling_method)

    hidden1_weights = tf.Variable(tf.random_normal(
        [cluster_size, hidden1_size],
        stddev=1 / math.sqrt(cluster_size)))
    tf.summary.histogram("hidden1_weights", hidden1_weights)
    activation = tf.matmul(activation, hidden1_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")
    else:
      hidden1_biases = tf.Variable(
          tf.random_normal(
              [hidden1_size], stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("hidden1_output", activation)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        **unused_params)

class LstmModel(models.BaseModel):

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

    ## Batch normalize the input
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)

    loss = 0.0
    with tf.variable_scope("RNN"):
      outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                         sequence_length=num_frames, 
                                         swap_memory=FLAGS.rnn_swap_memory,
                                         dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class BiLstmModel(models.BaseModel):

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

    ## Batch normalize the input
    fw_stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)
    bw_stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)

    loss = 0.0
    with tf.variable_scope("RNN"):
      outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw = fw_stacked_lstm, cell_bw = bw_stacked_lstm, 
                                         inputs = model_input,
                                         sequence_length=num_frames,
                                         swap_memory=FLAGS.rnn_swap_memory,
                                         dtype=tf.float32)
    state_fw, state_bw = states
    state = tf.concat([state_fw, state_bw], axis = 1)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class LstmParallelModel(models.BaseModel):

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
    number_of_layers = FLAGS.lstm_layers

    lstm_sizes = map(int, FLAGS.lstm_cells.split(","))
    feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
        FLAGS.feature_names, FLAGS.feature_sizes)
    sub_inputs = tf.split(model_input, feature_sizes, axis = 2)

    assert len(lstm_sizes) == len(feature_sizes), \
      "length of lstm_sizes (={}) != length of feature_sizes (={})".format( \
      len(lstm_sizes), len(feature_sizes))

    loss = 0.0
    outputs = []
    states = []
    for i in xrange(len(feature_sizes)):
      with tf.variable_scope("RNN%d" % i):
        sub_input = sub_inputs[i]
        lstm_size = lstm_sizes[i]
        ## Batch normalize the input
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
                [
                    tf.contrib.rnn.BasicLSTMCell(
                        lstm_size, forget_bias=1.0, state_is_tuple=False)
                    for _ in range(number_of_layers)
                    ],
                state_is_tuple=False)

        output, state = tf.nn.dynamic_rnn(stacked_lstm, sub_input,
                                         sequence_length=num_frames,
                                         swap_memory=FLAGS.rnn_swap_memory,
                                         dtype=tf.float32)
        outputs.append(output)
        states.append(state)

    # concat
    final_state = tf.concat(states, axis = 1)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=final_state,
        vocab_size=vocab_size,
        **unused_params)

class LstmPoolingModel(models.BaseModel):

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

    ## Batch normalize the input
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)

    loss = 0.0
    with tf.variable_scope("RNN"):
      outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                         sequence_length=num_frames,
                                         time_major=False,
                                         swap_memory=FLAGS.rnn_swap_memory,
                                         dtype=tf.float32)
      num_frames_matrix = tf.maximum(tf.cast(tf.expand_dims(num_frames, axis=1), dtype=tf.float32), tf.ones([batch_size, 1]))
      pooling_output = tf.reduce_sum(outputs, axis = 1) / num_frames_matrix

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=pooling_output,
        vocab_size=vocab_size,
        **unused_params)

class LstmWithMeanInputModel(models.BaseModel):

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

    ## Batch normalize the input
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)

    loss = 0.0
    with tf.variable_scope("RNN"):
      outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                         sequence_length=num_frames,
                                         time_major=False,
                                         dtype=tf.float32)

    mean_input = tf.reduce_mean(model_input, axis = 1)
    final_output = tf.concat([mean_input, state], axis = 1)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    predictions = aggregated_model().create_model(
        model_input=final_output,
        vocab_size=vocab_size,
        **unused_params)
    return predictions

class LstmWithPoolingModel(models.BaseModel):

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

    ## Batch normalize the input
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)

    loss = 0.0
    with tf.variable_scope("RNN"):
      outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                         sequence_length=num_frames,
                                         time_major=False,
                                         swap_memory=FLAGS.rnn_swap_memory,
                                         dtype=tf.float32)
      num_frames_matrix = tf.maximum(tf.cast(tf.expand_dims(num_frames, axis=1), dtype=tf.float32), tf.ones([batch_size, 1]))
      pooling_output = tf.reduce_sum(outputs, axis = 1) / num_frames_matrix
      final_output = tf.concat([pooling_output, state], axis = 1)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=final_output,
        vocab_size=vocab_size,
        **unused_params)

class GruPoolingModel(models.BaseModel):

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
    gru_size = FLAGS.gru_cells
    number_of_layers = FLAGS.gru_layers

    ## Batch normalize the input
    stacked_gru = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.GRUCell(gru_size)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)

    loss = 0.0
    with tf.variable_scope("RNN"):
      outputs, state = tf.nn.dynamic_rnn(stacked_gru, model_input,
                                         sequence_length=num_frames,
                                         time_major=False,
                                         swap_memory=FLAGS.rnn_swap_memory,
                                         dtype=tf.float32)
      num_frames_matrix = tf.maximum(tf.cast(tf.expand_dims(num_frames, axis=1), dtype=tf.float32), tf.ones([batch_size, 1]))
      pooling_output = tf.reduce_sum(outputs, axis = 1) / num_frames_matrix

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=pooling_output,
        vocab_size=vocab_size,
        **unused_params)

class GruWithPoolingModel(models.BaseModel):

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
    gru_size = FLAGS.gru_cells
    number_of_layers = FLAGS.gru_layers

    ## Batch normalize the input
    stacked_gru = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.GRUCell(gru_size)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)

    loss = 0.0
    with tf.variable_scope("RNN"):
      outputs, state = tf.nn.dynamic_rnn(stacked_gru, model_input,
                                         sequence_length=num_frames,
                                         time_major=False,
                                         swap_memory=FLAGS.rnn_swap_memory,
                                         dtype=tf.float32)
      num_frames_matrix = tf.maximum(tf.cast(tf.expand_dims(num_frames, axis=1), dtype=tf.float32), tf.ones([batch_size, 1]))
      pooling_output = tf.reduce_sum(outputs, axis = 1) / num_frames_matrix
      final_output = tf.concat([pooling_output, state], axis = 1)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=final_output,
        vocab_size=vocab_size,
        **unused_params)

class LstmDividedModel(models.BaseModel):

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

    with tf.device("/gpu:1"):
      ## Batch normalize the input
      stacked_lstm = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0, state_is_tuple=False)
                  for _ in range(number_of_layers)
                  ],
              state_is_tuple=False)
  
      loss = 0.0
    with tf.variable_scope("RNN"):
      outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                         sequence_length=num_frames,
                                         swap_memory=FLAGS.rnn_swap_memory,
                                         dtype=tf.float32)
  
    with tf.device("/gpu:0"):
      aggregated_model = getattr(video_level_models,
                                 FLAGS.video_level_classifier_model)
      predictions = aggregated_model().create_model(
          model_input=state,
          vocab_size=vocab_size,
          **unused_params)
    return predictions

class CnnKmaxModel(models.BaseModel):

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
    # Create a convolution + maxpool layer for each filter size
    filter_sizes = map(int, FLAGS.cnn_filter_sizes.split(","))
    filter_nums = map(int, FLAGS.cnn_filter_nums.split(","))
    pooling_k = FLAGS.cnn_pooling_k

    assert len(filter_sizes) == len(filter_nums), \
      "length of filter_sizes (={}) != length of filter_nums (={})".format( \
      len(filter_sizes), len(filter_nums))

    batch_size, max_frames, num_features = model_input.get_shape().as_list()

    with tf.variable_scope("CNN"):

      # set channel dimension to 1
      # cnn_input.shape = [batch, in_height, in_width, in_channels]
      cnn_input = tf.expand_dims(model_input, axis = 3)
            
      cnn_output = []
      for filter_size, filter_num in zip(filter_sizes, filter_nums):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
          # Convolution Layer
          # filter.shape = [filter_height, filter_width, in_channels, out_channels]
          filter_shape = [filter_size, num_features, 1, filter_num]

          W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
          b = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="b")

          tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(W))
          tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(b))

          # conv.shape = [batch, -1, 1, out_channels]
          conv = tf.nn.conv2d(cnn_input, W, strides=[1, 1, 1, 1], padding="VALID",name="conv")
          # add bias
          conv = tf.nn.bias_add(conv, b)
          if pooling_k == 1:
            _, conv_len, _, _= conv.shape.as_list()
            conv_kmax = tf.reduce_max(conv, axis = 1)
            conv_flat = tf.reshape(conv_kmax, [-1, filter_num])
          else:
            conv_kmax = tf.transpose(tf.squeeze(conv, axis = 2), perm = [0, 2, 1])
            conv_kmax, _ = tf.nn.top_k(conv_kmax, k = pooling_k, sorted = True)
            conv_flat = tf.reshape(conv_kmax, [-1, filter_num * pooling_k])
          cnn_output.append(conv_flat)
      cnn_output = tf.concat(cnn_output, axis = 1)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=conv_flat,
        vocab_size=vocab_size,
        **unused_params)

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
        vocab_size=vocab_size,
        **unused_params)

