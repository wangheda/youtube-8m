import math
import models
import tensorflow as tf
import utils
from tensorflow import flags
import tensorflow.contrib.slim as slim
FLAGS = flags.FLAGS

class MultiTaskDivergenceDeepCombineChainModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self, model_input, vocab_size, num_mixtures=None,
                   l2_penalty=1e-8, sub_scope="ddcc", original_input=None, 
                   dropout=False, keep_prob=None, noise_level=None,
                   num_frames=None, **unused_params):
    num_supports = FLAGS.num_supports
    num_models = FLAGS.divergence_model_count

    support_predictions = []
    for i in xrange(num_models):
      sub_prediction = self.sub_chain_model(model_input,vocab_size, num_mixtures, 
                                      l2_penalty, sub_scope+"%d"%i, original_input,
                                      dropout, keep_prob, noise_level)
      support_predictions.append(sub_prediction)
    support_predictions = tf.stack(support_predictions, axis=1)
    main_predictions = tf.reduce_mean(support_predictions, axis=1)
    return {"predictions": main_predictions, "support_predictions": support_predictions}

  def sub_chain_model(self, model_input, vocab_size, num_mixtures=None,
                      l2_penalty=1e-8, sub_scope="", original_input=None, 
                      dropout=False, keep_prob=None, noise_level=None,
                      num_frames=None,
                      **unused_params):
    num_supports = FLAGS.num_supports
    num_layers = FLAGS.deep_chain_layers
    relu_cells = FLAGS.deep_chain_relu_cells
    relu_type = FLAGS.deep_chain_relu_type
    use_length = FLAGS.deep_chain_use_length

    input_list = [model_input]
    mean_activation = slim.fully_connected(
        model_input,
        relu_cells,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope=sub_scope+"relu-mean")
    mean_relu = tf.nn.relu(mean_activation)

    if noise_level is not None:
      print "adding noise to sub_relu, level = ", noise_level
      mean_relu = mean_relu + tf.random_normal(tf.shape(mean_relu), mean=0.0, stddev=noise_level)

    mean_relu_norm = tf.nn.l2_normalize(mean_relu, dim=1)

    next_input = mean_relu_norm
    support_predictions = []
    for layer in xrange(num_layers):
      sub_prediction = self.sub_model(next_input, vocab_size, sub_scope=sub_scope+"prediction-%d"%layer, dropout=dropout, keep_prob=keep_prob, noise_level=noise_level)
      sub_activation = slim.fully_connected(
          sub_prediction,
          relu_cells,
          activation_fn=None,
          weights_regularizer=slim.l2_regularizer(l2_penalty),
          scope=sub_scope+"relu-%d"%layer)
      sub_relu = tf.nn.relu(sub_activation)

      if noise_level is not None:
        print "adding noise to sub_relu, level = ", noise_level
        sub_relu = sub_relu + tf.random_normal(tf.shape(sub_relu), mean=0.0, stddev=noise_level)

      relu_norm = tf.nn.l2_normalize(sub_relu, dim=1)
      next_input = tf.concat([next_input, relu_norm], axis=1)
      support_predictions.append(sub_prediction)

    main_predictions = self.sub_model(next_input, vocab_size, sub_scope=sub_scope+"-main")
    support_predictions = tf.reduce_mean(tf.stack(support_predictions, axis=0), axis=0)

    ratio = min(FLAGS.support_loss_percent, 0.2)
    final_predictions = main_predictions * (1.0 - ratio) + support_predictions * ratio
    return final_predictions

  def sub_model(self, model_input, vocab_size, num_mixtures=None, 
                l2_penalty=1e-8, sub_scope="", 
                dropout=False, keep_prob=None, noise_level=None,
                **unused_params):
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    if dropout:
      model_input = tf.nn.dropout(model_input, keep_prob=keep_prob)

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

