import math
import models
import tensorflow as tf
import utils
from tensorflow import flags
import tensorflow.contrib.slim as slim
FLAGS = flags.FLAGS

class AttentionMoeModel(models.BaseModel):

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

    original_input = tf.nn.l2_normalize(original_input, dim=1)
    model_input_list = tf.unstack(model_input, axis=2)
    
    relu_units = [self.relu(original_input, num_relu, sub_scope="input")]
    i = 0
    for mi in model_input_list:
      relu_units.append(self.relu(mi, num_relu, sub_scope="sub"+str(i)))
      i += 1

    gate_activations = slim.fully_connected(
        tf.concat(relu_units, axis=1),
        num_methods,
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gate")
    gate = tf.nn.softmax(gate_activations)
    output = tf.einsum("ijk,ik->ij", model_input, gate)
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

