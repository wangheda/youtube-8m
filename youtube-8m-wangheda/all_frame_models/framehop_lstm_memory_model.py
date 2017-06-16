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

class FramehopLstmMemoryModel(models.BaseModel):
  """Classifier chain model of lstm memory"""

  def lstm(self, model_input, vocab_size, num_frames, sub_scope="", 
           feature_names=None, feature_sizes=None, **unused_params):
    number_of_layers = FLAGS.lstm_layers
    lstm_sizes = map(int, FLAGS.lstm_cells.split(","))
    if feature_names is None:
        feature_names = FLAGS.feature_names
    if feature_sizes is None:
        feature_sizes = FLAGS.feature_sizes
    feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
        feature_names, feature_sizes)
    sub_inputs = [tf.nn.l2_normalize(x, dim=2) for x in tf.split(model_input, feature_sizes, axis = 2)]

    print model_input
    assert len(lstm_sizes) == len(feature_sizes), \
      "length of lstm_sizes (={}) != length of feature_sizes (={})".format( \
      len(lstm_sizes), len(feature_sizes))

    states = []
    outputs = []
    for i in xrange(len(feature_sizes)):
      with tf.variable_scope(sub_scope+"RNN%d" % i):
        print sub_scope + "RNN%d"%i, lstm_sizes
        print sub_inputs
        sub_input = sub_inputs[i]
        lstm_size = lstm_sizes[i]
        ## Batch normalize the input
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
                [
                    tf.contrib.rnn.BasicLSTMCell(
                        lstm_size, forget_bias=1.0, state_is_tuple=True)
                    for _ in range(number_of_layers)
                    ],
                state_is_tuple=True)
        output, state = tf.nn.dynamic_rnn(stacked_lstm, sub_input,
                                         sequence_length=num_frames,
                                         swap_memory=FLAGS.rnn_swap_memory,
                                         dtype=tf.float32)
        states.extend(map(lambda x: x.c, state))
        outputs.append(output)
    final_state = tf.concat(states, axis = 1)
    final_output = tf.concat(outputs, axis = 2)
    return final_state, final_output

  def create_model(self, model_input, vocab_size, num_mixtures=None,
                   l2_penalty=1e-8, sub_scope="", original_input=None, 
                   dropout=False, keep_prob=None, noise_level=None,
                   num_frames=None,
                   **unused_params):
    num_supports = FLAGS.num_supports
    num_layers = FLAGS.deep_chain_layers
    relu_cells = FLAGS.deep_chain_relu_cells
    relu_type = FLAGS.deep_chain_relu_type
    use_length = FLAGS.deep_chain_use_length

    lstm_states = []
    if use_length:
      print "using length as feature"
      additional_features.append(self.get_length_code(num_frames))

    lstm_input, lstm_frames = self.resolution(model_input, num_frames, 1)
    lstm_state, lstm_output = self.lstm(lstm_input, vocab_size, num_frames=lstm_frames, sub_scope="lstm%d"%0)
    lstm_states.append(lstm_state)

    support_predictions = []
    for layer in xrange(num_layers):
      lstm_input, lstm_frames = self.resolution(lstm_output, num_frames, 2)
      lstm_state, lstm_output = self.lstm(lstm_input, vocab_size, num_frames=lstm_frames, 
          feature_sizes=FLAGS.lstm_cells, sub_scope="lstm%d"%(layer+1))
      lstm_states.append(lstm_state)

    final_states = tf.concat(lstm_states, axis=1)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=final_states,
        original_input=model_input,
        vocab_size=vocab_size,
        num_frames=num_frames,
        **unused_params)

  def get_length_code(self, num_frames):
    code_0 = tf.cast(num_frames <= 60, dtype=tf.int32)
    code_1 = tf.cast(num_frames > 60, dtype=tf.int32) * tf.cast(num_frames <= 120, dtype=tf.int32)
    code_2 = tf.cast(num_frames > 120, dtype=tf.int32) * tf.cast(num_frames <= 180, dtype=tf.int32)
    code_3 = tf.cast(num_frames > 180, dtype=tf.int32) * tf.cast(num_frames <= 240, dtype=tf.int32)
    code_4 = tf.cast(num_frames > 240, dtype=tf.int32)
    codes = map(lambda x: tf.expand_dims(x, dim=1), [code_0, code_1, code_2, code_3, code_4])
    length_code = tf.cast(tf.concat(codes, axis=1), dtype=tf.float32)
    return length_code

  def resolution(self, model_input_raw, num_frames, resolution, method="SELECT"):
    frame_dim = len(model_input_raw.get_shape()) - 2
    feature_dim = len(model_input_raw.get_shape()) - 1
    max_frames = model_input_raw.get_shape().as_list()[frame_dim]
    num_features = model_input_raw.get_shape().as_list()[feature_dim]
    if resolution > 1:
      new_max_frames = max_frames / resolution
      cut_frames = new_max_frames * resolution
      model_input_raw = model_input_raw[:, :cut_frames, :]
      model_input_raw = tf.reshape(model_input_raw, shape=[-1,new_max_frames,resolution,num_features])
      if method == "MEAN":
        model_input_raw = tf.reduce_mean(model_input_raw, axis=2)
      elif method == "MAX":
        model_input_raw = tf.reduce_max(model_input_raw, axis=2)
      elif method == "SELECT":
        model_input_raw = model_input_raw[:,:,resolution-1,:]
      model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)
      num_frames = num_frames / resolution
    else:
      model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)
    return model_input, num_frames

