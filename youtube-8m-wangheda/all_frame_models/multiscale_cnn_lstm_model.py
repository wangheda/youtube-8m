import math
import models
import tensorflow as tf
import numpy as np
import utils
from tensorflow import flags
import tensorflow.contrib.slim as slim
FLAGS = flags.FLAGS

class MultiscaleCnnLstmModel(models.BaseModel):

  def cnn(self, 
          model_input, 
          l2_penalty=1e-8, 
          num_filters = [1024, 1024, 1024],
          filter_sizes = [1,2,3], 
          sub_scope="",
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
      sub_filter = tf.get_variable(sub_scope+"cnn-filter-len%d"%fs, 
                       shape=[num_features*fs, nf], dtype=tf.float32, 
                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1), 
                       regularizer=tf.contrib.layers.l2_regularizer(l2_penalty))
      cnn_outputs.append(tf.einsum("ijk,kl->ijl", sub_input, sub_filter))

    cnn_output = tf.concat(cnn_outputs, axis=2)
    cnn_output = slim.batch_norm(
        cnn_output,
        center=True,
        scale=True,
        is_training=FLAGS.is_training,
        scope=sub_scope+"cluster_bn")
    return cnn_output

  def moe(self,model_input,
          vocab_size,
          num_mixtures=None,
          l2_penalty=1e-8,
          scopename="",
          **unused_params):

    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates"+scopename)
    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts"+scopename)

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

  def rnn(self, model_input, lstm_size, num_frames,
          sub_scope="", **unused_params):

    cell = tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=1.0, 
                                        state_is_tuple=True)
    with tf.variable_scope("RNN-"+sub_scope):
      outputs, state = tf.nn.dynamic_rnn(cell, model_input,
                                         sequence_length=num_frames,
                                         swap_memory=True,
                                         dtype=tf.float32)
    # return final memory
    return state.c

  def create_model(self, model_input, vocab_size, num_frames, 
                   l2_penalty=1e-8, **unused_params):

    num_layers = FLAGS.multiscale_cnn_lstm_layers
    lstm_size = int(FLAGS.lstm_cells)
    pool_size=2
    num_filters=[256,256,512]
    filter_sizes=[1,2,3]
    features_size = sum(num_filters)

    sub_predictions = []
    cnn_input = model_input

    cnn_max_frames = model_input.get_shape().as_list()[1]

    for layer in range(num_layers):
      cnn_output = self.cnn(cnn_input, num_filters=num_filters, filter_sizes=filter_sizes, sub_scope="cnn%d"%(layer+1))
      cnn_output_relu = tf.nn.relu(cnn_output)

      lstm_memory = self.rnn(cnn_output_relu, lstm_size, num_frames, sub_scope="rnn%d"%(layer+1))
      sub_prediction = self.moe(lstm_memory, vocab_size, scopename="moe%d"%(layer+1))
      sub_predictions.append(sub_prediction)

      cnn_max_frames /= pool_size
      max_pooled_cnn_output = tf.reduce_max(
          tf.reshape(
              cnn_output_relu[:, :cnn_max_frames*2, :], 
              [-1, cnn_max_frames, pool_size, features_size]
          ), axis=2)

      # for the next cnn layer
      cnn_input = max_pooled_cnn_output
      num_frames = tf.maximum(num_frames/pool_size, 1)

    support_predictions = tf.concat(sub_predictions, axis=1)
    predictions = tf.add_n(sub_predictions) / len(sub_predictions)

    return {"predictions": predictions, 
            "support_predictions": support_predictions}

