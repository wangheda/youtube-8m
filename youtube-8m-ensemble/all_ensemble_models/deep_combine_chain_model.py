import math
import models
import tensorflow as tf
import utils
from tensorflow import flags
import tensorflow.contrib.slim as slim
FLAGS = flags.FLAGS

class DeepCombineChainModel(models.BaseModel):

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   sub_scope="",
                   original_input=None, 
                   **unused_params):

    num_methods = model_input.get_shape().as_list()[-1]
    num_features = model_input.get_shape().as_list()[-2]
    num_mixtures = FLAGS.moe_num_mixtures
    attention_matrix_rank = FLAGS.attention_matrix_rank
    relu_cells = FLAGS.deep_chain_relu_cells

    # mean_output
    mean_output = tf.reduce_mean(model_input, axis=2)
    mean_relu = slim.fully_connected(
        mean_output,
        relu_cells,
        activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        reuse=True,
        scope=sub_scope+"transform")

    permuted_input = tf.transpose(model_input, perm=[0,2,1])
    input_relu = slim.fully_connected(
        permuted_input,
        relu_cells,
        activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        reuse=True,
        scope=sub_scope+"transform")
    input_relu_list = tf.unstack(input_relu, axis=1)

    predictions = []
    discounted_prediction = mean_output
    prev_relu = mean_relu
    for i in xrange(num_methods):
      cur_relu = input_relu_list[i]
      cur_prediction = self.sub_moe(
          model_input=tf.concat([prev_relu, cur_relu], axis=1),
          vocab_size=vocab_size,
          sub_scope="predict-%d"%i)
      discounted_prediction = (discounted_prediction + cur_prediction) / 2
      # if not the last one
      if i + 1 < num_methods:
        predictions.append(cur_prediction)
        new_relu = slim.fully_connected(
            cur_prediction,
            relu_cells,
            activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            reuse=True,
            scope=sub_scope+"compress")

    final_prediction = discounted_prediction
    support_prediction = tf.reduce_mean(tf.stack(predictions, axis=0), axis=0)
    return {"predictions": final_prediction,
            "support_predictions": support_prediction}

  def sub_moe(self, model_input, vocab_size, num_mixtures=None, 
                l2_penalty=1e-8, sub_scope="", 
                **unused_params):

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

