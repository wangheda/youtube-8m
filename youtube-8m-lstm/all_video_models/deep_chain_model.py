import math
import models
import tensorflow as tf
import utils
from tensorflow import flags
import tensorflow.contrib.slim as slim
FLAGS = flags.FLAGS

class DeepChainModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self, model_input, vocab_size, num_mixtures=None,
                   l2_penalty=1e-8, sub_scope="", original_input=None, 
                   num_frames=None, **unused_params):
    num_supports = FLAGS.num_supports
    num_layers = FLAGS.deep_chain_layers
    relu_cells = FLAGS.deep_chain_relu_cells
    use_length = FLAGS.deep_chain_use_length

    if use_length:
      model_input = tf.concat([model_input, self.get_length_code(num_frames)], axis=1)

    next_input = model_input
    support_predictions = []
    for layer in xrange(num_layers):
      sub_prediction = self.sub_model(next_input, vocab_size, sub_scope=sub_scope+"prediction-%d"%layer)
      sub_relu = slim.fully_connected(
          sub_prediction,
          relu_cells,
          activation_fn=tf.nn.relu,
          weights_regularizer=slim.l2_regularizer(l2_penalty),
          scope=sub_scope+"relu-%d"%layer)
      relu_norm = tf.nn.l2_normalize(sub_relu, dim=1)
      next_input = tf.concat([model_input, relu_norm], axis=1)
      support_predictions.append(sub_prediction)
    main_predictions = self.sub_model(next_input, vocab_size, sub_scope=sub_scope+"-main")
    support_predictions = tf.concat(support_predictions, axis=1)
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

  def get_length_code(self, num_frames):
    code_0 = tf.cast(num_frames <= 60, dtype=tf.int32)
    code_1 = tf.cast(num_frames > 60, dtype=tf.int32) * tf.cast(num_frames <= 120, dtype=tf.int32)
    code_2 = tf.cast(num_frames > 120, dtype=tf.int32) * tf.cast(num_frames <= 180, dtype=tf.int32)
    code_3 = tf.cast(num_frames > 180, dtype=tf.int32) * tf.cast(num_frames <= 240, dtype=tf.int32)
    code_4 = tf.cast(num_frames > 240, dtype=tf.int32)
    codes = map(lambda x: tf.expand_dims(x, dim=1), [code_0, code_1, code_2, code_3, code_4])
    length_code = tf.cast(tf.concat(codes, axis=1), dtype=tf.float32)
    return length_code
