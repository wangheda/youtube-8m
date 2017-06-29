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

class LstmAuxlossDeepCombineChainModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, num_mixtures=None,
                   l2_penalty=1e-8, sub_scope="", original_input=None, **unused_params):
    """Creates a model that use different times of output of lstm
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
    """
    number_of_layers = FLAGS.lstm_layers
    deep_chain_layers = FLAGS.deep_chain_layers
    relu_cells = FLAGS.deep_chain_relu_cells
    batch_size = tf.shape(model_input)[0]

    lstm_sizes = map(int, FLAGS.lstm_cells.split(","))
    feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
        FLAGS.feature_names, FLAGS.feature_sizes)
    sub_inputs = [tf.nn.l2_normalize(x, dim=2) for x in tf.split(model_input, feature_sizes, axis = 2)]

    assert len(lstm_sizes) == len(feature_sizes), \
      "length of lstm_sizes (={}) != length of feature_sizes (={})".format( \
      len(lstm_sizes), len(feature_sizes))

    outputs = []
    for i in xrange(len(feature_sizes)):
      with tf.variable_scope("RNN%d" % i):
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
                                         time_major=False,
                                         dtype=tf.float32)
        outputs.append(output)

    # generate time to get frame output
    num_layers = deep_chain_layers + 1
    timed_num_frames = []
    print "num_layers =", num_layers
    unit_num_frames = num_frames / num_layers
    for i in xrange(deep_chain_layers):
      timed_num_frames.append(unit_num_frames * (i+1))
    timed_num_frames.append(num_frames)

    # generate frame outputs from certain time points (num_frames - 1 to turn length into index (base 0))
    timed_outputs = []
    for sub_num_frames in timed_num_frames:
      frame_outputs = []
      for output in outputs:
        frame_index = tf.stack([tf.range(batch_size), tf.maximum(sub_num_frames-1,0)], axis=1)
        frame_output = tf.gather_nd(output, frame_index)
        frame_outputs.append(frame_output)
      timed_outputs.append(tf.concat(frame_outputs, axis=1))

    # deep combine model
    predictions = []
    relu_layers = []
    for layer in xrange(num_layers):
      if relu_layers:
        next_input = tf.concat([timed_outputs[layer]] + relu_layers, axis=1)
      else:
        next_input = timed_outputs[layer]
      sub_prediction = self.sub_model(next_input, vocab_size, sub_scope=sub_scope+"prediction-%d"%layer)
      predictions.append(sub_prediction)

      sub_relu = slim.fully_connected(
          sub_prediction,
          relu_cells,
          activation_fn=tf.nn.relu,
          weights_regularizer=slim.l2_regularizer(l2_penalty),
          scope=sub_scope+"relu-%d"%layer)
      if layer + 1 < num_layers:
        relu_norm = tf.nn.l2_normalize(sub_relu, dim=1)
        relu_layers.append(relu_norm)

    main_predictions = predictions[-1]
    support_predictions = tf.concat(predictions[:-1], axis=1)
    return {"predictions": main_predictions, "support_predictions": support_predictions}

  def sub_model(self, model_input, vocab_size, num_mixtures=None, 
                l2_penalty=1e-8, sub_scope="", **unused_params):
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates-"+sub_scope)
    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts-"+sub_scope)

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return final_probabilities

