import math
import models
import tensorflow as tf
import utils
from tensorflow import flags
import tensorflow.contrib.slim as slim
FLAGS = flags.FLAGS

class AttentionLinearModel(models.BaseModel):

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
    weight_var = tf.get_variable("ensemble_weight",
        shape=[num_mixtures, num_methods],
        regularizer=slim.l2_regularizer(l2_penalty))

    # weight
    gated_weight = tf.einsum("ij,jk->ik", gate_activations, weight_var)
    weight = tf.nn.softmax(gated_weight)
    
    # weighted output
    output = tf.einsum("ik,ijk->ij", weight, model_input)
    return {"predictions": output}
