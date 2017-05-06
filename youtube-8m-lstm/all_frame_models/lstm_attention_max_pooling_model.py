import math
import models
import tensorflow as tf
import numpy as np
import utils
from tensorflow import flags
import tensorflow.contrib.slim as slim
FLAGS = flags.FLAGS

class LstmAttentionMaxPoolingModel(models.BaseModel):
  """Max pooling over temporal weighted sums (attention) of lstm outputs."""

  def create_model(self, model_input, vocab_size, num_frames, 
                   num_mixtures=None, l2_penalty=1e-8, sub_scope="", 
                   original_input=None, **unused_params):
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
    num_attentions = FLAGS.lstm_attentions

    batch_size, max_frames, num_features = model_input.get_shape().as_list()
    mask = tf.sequence_mask(lengths=num_frames, maxlen=max_frames, dtype=tf.float32)

    ## Batch normalize the input
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
        [
            tf.contrib.rnn.BasicLSTMCell(
                lstm_size, forget_bias=1.0, state_is_tuple=True)
            for _ in range(number_of_layers)
            ],
        state_is_tuple=True)

    with tf.variable_scope("RNN"):
      outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                           sequence_length=num_frames,
                           swap_memory=FLAGS.rnn_swap_memory,
                           dtype=tf.float32)

    attention_activations = slim.fully_connected(
        tf.concat([model_input, outputs], axis=2),
        num_attentions,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="attention-"+sub_scope)

    # Batch x #Attentions x #Frames
    attention_weights = tf.einsum("ijk,ij->ikj", tf.nn.softmax(attention_activations, dim=1), mask)
    attention_weights = attention_weights / tf.reduce_sum(attention_weights, axis=2, keep_dims=True)

    # Batch x #Attentions x #Features
    attention_outputs = tf.einsum("ijk,ilj->ilk", outputs, attention_weights)
    moe_predictions = self.sub_moe(attention_outputs, vocab_size, sub_scope="sub-moe")
    predictions = tf.reshape(moe_predictions, [-1, num_attentions, vocab_size])
    max_predictions = tf.reduce_max(predictions, axis=1)

    return {"predictions": max_predictions}

  def sub_moe(self, model_input, vocab_size, num_mixtures=None,
              l2_penalty=1e-8, sub_scope="",  **unused_params):
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
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch, [-1, vocab_size])
    return final_probabilities

