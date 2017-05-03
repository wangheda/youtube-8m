import math
import models
import tensorflow as tf
import utils
from tensorflow import flags
import tensorflow.contrib.slim as slim
FLAGS = flags.FLAGS

class InputMoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

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

    original_input = tf.nn.l2_normalize(original_input, dim=1)
    gate_activations = slim.fully_connected(
        original_input,
        num_methods,
        activation_fn=tf.nn.softmax,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates"+sub_scope)

    output = tf.einsum("ijk,ik->ij", model_input, gate_activations)
    return {"predictions": output}
