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

class DistillchainLstmParallelFinaloutputModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames,
                   distillation_predictions=None,
                   l2_penalty=1e-8,
                   **unused_params):
  
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
    assert distillation_predictions is not None, "distillation feature must be used"

    relu_cells = FLAGS.distillchain_relu_cells
    number_of_layers = FLAGS.lstm_layers

    distill_relu = slim.fully_connected(
          distillation_predictions,
          relu_cells,
          activation_fn=tf.nn.relu,
          weights_regularizer=slim.l2_regularizer(l2_penalty),
          scope="distillrelu")
    distill_norm = tf.nn.l2_normalize(distill_relu, dim=1)

    lstm_sizes = map(int, FLAGS.lstm_cells.split(","))
    feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
        FLAGS.feature_names, FLAGS.feature_sizes)
    sub_inputs = [tf.nn.l2_normalize(x, dim=2) for x in tf.split(model_input, feature_sizes, axis = 2)]

    assert len(lstm_sizes) == len(feature_sizes), \
      "length of lstm_sizes (={}) != length of feature_sizes (={})".format( \
      len(lstm_sizes), len(feature_sizes))

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
                        lstm_size, forget_bias=1.0, state_is_tuple=True)
                    for _ in range(number_of_layers)
                    ],
                state_is_tuple=True)

        output, state = tf.nn.dynamic_rnn(stacked_lstm, sub_input,
                                         sequence_length=num_frames,
                                         swap_memory=FLAGS.rnn_swap_memory,
                                         dtype=tf.float32)
        outputs.append(output)
        states.extend(map(lambda x: x.h, state))

    # concat
    final_state = tf.concat(states + [distill_norm], axis = 1)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=final_state,
        original_input=model_input,
        vocab_size=vocab_size,
        **unused_params)

