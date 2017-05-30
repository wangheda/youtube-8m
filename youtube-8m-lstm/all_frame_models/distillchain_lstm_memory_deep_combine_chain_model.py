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

class DistillchainLstmMemoryDeepCombineChainModel(models.BaseModel):
  """Classifier chain model of lstm memory"""

  def create_model(self, model_input, vocab_size, num_frames, 
                   l2_penalty=1e-8, sub_scope="", original_input=None, 
                   distillation_predictions=None,
                   **unused_params):

    assert distillation_predictions is not None, "distillation feature must be used"

    distillchain_relu_cells = FLAGS.distillchain_relu_cells
    lstm_size = int(FLAGS.lstm_cells)
    number_of_layers = FLAGS.lstm_layers
    num_supports = FLAGS.num_supports
    num_layers = FLAGS.deep_chain_layers
    relu_cells = FLAGS.deep_chain_relu_cells
    max_frames = model_input.get_shape().as_list()[1]
    relu_layers = []
    support_predictions = []
    
    # distill predictions
    distill_relu = slim.fully_connected(
          distillation_predictions,
          distillchain_relu_cells,
          activation_fn=tf.nn.relu,
          weights_regularizer=slim.l2_regularizer(l2_penalty),
          scope="distill-relu")
    distill_norm = tf.nn.l2_normalize(distill_relu, dim=1)
    relu_layers.append(distill_norm)

    # mean input
    mask = tf.sequence_mask(num_frames, maxlen=max_frames, dtype=tf.float32)
    mean_input = tf.einsum("ijk,ij->ik", model_input, mask) \
          / tf.expand_dims(tf.cast(num_frames, dtype=tf.float32), dim=1)
    mean_relu = slim.fully_connected(
          mean_input,
          relu_cells,
          activation_fn=tf.nn.relu,
          weights_regularizer=slim.l2_regularizer(l2_penalty),
          scope=sub_scope+"mean-relu")
    mean_relu_norm = tf.nn.l2_normalize(mean_relu, dim=1)
    relu_layers.append(mean_relu_norm)

    lstm_output = self.sub_lstm(model_input, num_frames, lstm_size, number_of_layers, sub_scope="lstm-%d"%0)
    normalized_lstm_output = tf.nn.l2_normalize(lstm_output, dim=1)
    next_input = tf.concat([normalized_lstm_output] + relu_layers, axis=1)

    for layer in xrange(num_layers):
      sub_prediction = self.sub_moe(next_input, vocab_size, sub_scope=sub_scope+"prediction-%d"%layer)
      support_predictions.append(sub_prediction)

      sub_relu = slim.fully_connected(
          sub_prediction,
          relu_cells,
          activation_fn=tf.nn.relu,
          weights_regularizer=slim.l2_regularizer(l2_penalty),
          scope=sub_scope+"relu-%d"%layer)
      relu_norm = tf.nn.l2_normalize(sub_relu, dim=1)
      relu_layers.append(relu_norm) 

      lstm_output = self.sub_lstm(model_input, num_frames, lstm_size, number_of_layers, sub_scope="lstm-%d"%(layer+1))
      normalized_lstm_output = tf.nn.l2_normalize(lstm_output, dim=1)

      next_input = tf.concat([normalized_lstm_output] + relu_layers, axis=1)

    main_predictions = self.sub_moe(next_input, vocab_size, sub_scope=sub_scope+"-main")
    support_predictions = tf.concat(support_predictions, axis=1)
    return {"predictions": main_predictions, "support_predictions": support_predictions}

  def sub_lstm(self, model_input, num_frames, lstm_size, number_of_layers, sub_scope=""):
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=True)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=True)

    loss = 0.0
    with tf.variable_scope(sub_scope+"-RNN"):
      outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                         sequence_length=num_frames, 
                                         swap_memory=FLAGS.rnn_swap_memory,
                                         dtype=tf.float32)
      final_state = tf.concat(map(lambda x: x.c, state), axis = 1)
    return final_state

  def sub_moe(self, model_input, vocab_size, num_mixtures=None, 
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
