import math
import models
import tensorflow as tf
import utils
from tensorflow import flags
import tensorflow.contrib.slim as slim
FLAGS = flags.FLAGS

class Stage2LogisticModel(models.BaseModel):
  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, original_input=None, **unused_params):
    output = tf.nn.sigmoid(model_input + slim.fully_connected(
        model_input, vocab_size, activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty)))
    return {"predictions": output}

