import math
import models
import tensorflow as tf
import utils
from tensorflow import flags
import tensorflow.contrib.slim as slim
FLAGS = flags.FLAGS

class LinearRegressionModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, original_input=None, **unused_params):
    """Creates a linear regression model.

    Args:
      model_input: 'batch' x 'num_features' x 'num_methods' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = slim.fully_connected(
        model_input, 1, activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    output = tf.squeeze(output, axis=2)
    return {"predictions": output}

