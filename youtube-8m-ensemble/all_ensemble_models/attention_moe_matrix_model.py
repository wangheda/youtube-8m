import math
import models
import tensorflow as tf
import utils
from tensorflow import flags
import tensorflow.contrib.slim as slim
FLAGS = flags.FLAGS

class AttentionMoeMatrixModel(models.BaseModel):

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   sub_scope="",
                   original_input=None, 
                   **unused_params):

    num_relu = FLAGS.attention_relu_cells
    num_methods = model_input.get_shape().as_list()[-1]
    num_features = model_input.get_shape().as_list()[-2]
    num_mixtures = FLAGS.moe_num_mixtures
    attention_matrix_rank = FLAGS.attention_matrix_rank

    # gating coefficients
    mean_input = tf.reduce_mean(model_input, axis=2)
    std_input = tf.reduce_sum(
        tf.square(model_input - tf.expand_dims(mean_input, dim=2)), 
        axis=2) / (num_methods - 1)

    # relu
    original_relu = self.relu(original_input, num_relu, sub_scope="origin")
    mean_relu = self.relu(mean_input, num_relu, sub_scope="mean")
    std_relu = self.relu(std_input, num_relu, sub_scope="std")

    # normalize
    original_relu = tf.nn.l2_normalize(original_relu)
    mean_relu = tf.nn.l2_normalize(mean_relu)
    std_relu = tf.nn.l2_normalize(std_relu)

    ## batch_size x moe_num_mixtures
    gate_activations = slim.fully_connected(
        tf.concat([original_relu, mean_relu, std_relu], axis=1),
        num_mixtures,
        activation_fn=tf.nn.softmax,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates"+sub_scope)

    # matrix
    weight_x = tf.get_variable("ensemble_weightx", 
        shape=[num_mixtures, num_features, attention_matrix_rank],
        regularizer=slim.l2_regularizer(l2_penalty))
    weight_y = tf.get_variable("ensemble_weighty", 
        shape=[num_mixtures, attention_matrix_rank, num_methods],
        regularizer=slim.l2_regularizer(l2_penalty))
    ## moe_num_mixtures x num_features x num_methods
    weight_xy = tf.einsum("ijl,ilk->ijk", weight_x, weight_y)

    # weight
    gated_weight_xy = tf.einsum("ij,jkl->ikl", gate_activations, weight_xy)
    weight = tf.nn.softmax(gated_weight_xy)
    
    # weighted output
    output = tf.reduce_sum(weight * model_input, axis=2)
    return {"predictions": output}

  def relu(self, model_input, relu_cells, 
               l2_penalty=1e-8, sub_scope=""):
    sub_activation = slim.fully_connected(
        model_input,
        relu_cells,
        activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="relu-"+sub_scope)
    return sub_activation

