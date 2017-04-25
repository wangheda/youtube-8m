import math
import models
import tensorflow as tf
import utils
from tensorflow import flags
import tensorflow.contrib.slim as slim
FLAGS = flags.FLAGS

class MeanModel(models.BaseModel):
  """Mean model."""

  def create_model(self, model_input, **unused_params):
    """Creates a logistic model.

      model_input: 'batch' x 'num_features' x 'num_methods' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = tf.reduce_mean(model_input, axis=2)
    return {"predictions": output}

