import math
import models
import tensorflow as tf
import utils
from tensorflow import flags
import tensorflow.contrib.slim as slim
FLAGS = flags.FLAGS

class MatrixRegressionModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, original_input=None, **unused_params):
    """Creates a matrix regression model.

    Args:
      model_input: 'batch' x 'num_features' x 'num_methods' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    num_features = model_input.get_shape().as_list()[-2]
    num_methods = model_input.get_shape().as_list()[-1]

    weight1d = tf.get_variable("ensemble_weight1d", 
        shape=[num_methods],
        regularizer=slim.l2_regularizer(l2_penalty))
    weight2d = tf.get_variable("ensemble_weight2d", 
        shape=[num_features, num_methods],
        regularizer=slim.l2_regularizer(10 * l2_penalty))
    weight = tf.nn.softmax(tf.einsum("ij,j->ij", weight2d, weight1d), dim=-1)
    output = tf.einsum("ijk,jk->ij", model_input, weight)
    return {"predictions": output}

