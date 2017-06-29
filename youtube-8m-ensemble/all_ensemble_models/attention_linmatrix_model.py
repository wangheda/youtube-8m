import math
import models
import tensorflow as tf
import utils
from tensorflow import flags
import tensorflow.contrib.slim as slim
FLAGS = flags.FLAGS

class AttentionLinmatrixModel(models.BaseModel):

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

    # gating coefficients
    original_input = tf.nn.l2_normalize(original_input, dim=1)
    mean_output = tf.reduce_mean(model_input, axis=2)
    ## batch_size x moe_num_mixtures
    gate_activations = slim.fully_connected(
        tf.concat([original_input, mean_output], axis=1),
        num_mixtures,
        activation_fn=tf.nn.softmax,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates"+sub_scope)

    # matrix
    b = tf.get_variable("ensemble_bias", shape=[num_mixtures,1,1])
    weight_var = tf.get_variable("ensemble_weight",
        shape=[num_mixtures, 1, num_methods],
        regularizer=slim.l2_regularizer(l2_penalty))
    weight_x = tf.get_variable("ensemble_weightx", 
        shape=[num_mixtures, num_features, attention_matrix_rank],
        regularizer=slim.l2_regularizer(l2_penalty))
    weight_y = tf.get_variable("ensemble_weighty", 
        shape=[num_mixtures, attention_matrix_rank, num_methods],
        regularizer=slim.l2_regularizer(l2_penalty))
    ## moe_num_mixtures x num_features x num_methods
    weight_xy = tf.einsum("ijl,ilk->ijk", weight_x, weight_y) + weight_var + b

    # weight
    gated_weight_xy = tf.einsum("ij,jkl->ikl", gate_activations, weight_xy)
    weight = tf.nn.softmax(gated_weight_xy)
    
    # weighted output
    output = tf.reduce_sum(weight * model_input, axis=2)
    return {"predictions": output}
