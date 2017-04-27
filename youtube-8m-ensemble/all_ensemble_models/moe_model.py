import math
import models
import tensorflow as tf
import utils
from tensorflow import flags
import tensorflow.contrib.slim as slim
FLAGS = flags.FLAGS

class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   sub_scope="",
                   original_input=None, 
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_methods = model_input.get_shape().as_list()[-1]
    num_features = model_input.get_shape().as_list()[-2]

    flat_input = tf.reshape(model_input, shape=[-1,num_features * num_methods])

    tensor_weight = tf.get_variable("tensor_weight",
        shape=[num_features, num_methods, num_methods],
        regularizer=slim.l2_regularizer(l2_penalty))
    tensor_bias = tf.get_variable("tensor_bias",
        shape=[num_features, num_methods],
        initializer=tf.zeros_initializer(),
        regularizer=slim.l2_regularizer(l2_penalty))

    gate_activations = tf.einsum("ijk,jkl->ijl", model_input, tensor_weight) \
        + tf.expand_dims(tensor_bias, dim=0)

    output = tf.reduce_sum(model_input * tf.nn.softmax(gate_activations), axis=2)
    return {"predictions": output}
