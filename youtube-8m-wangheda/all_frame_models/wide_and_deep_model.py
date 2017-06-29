import sys
import models
import model_utils
import math
import numpy as np
import video_level_models
import frame_level_models
import tensorflow as tf
import utils
import tensorflow.contrib.slim as slim
from tensorflow import flags
FLAGS = flags.FLAGS

class WideAndDeepModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, l2_penalty=1e-8, **unused_params):
    """
    A super model that combine one or more models
    """
    models = FLAGS.wide_and_deep_models
    outputs = []
    for model_name in map(lambda x: x.strip(), models.split(",")):
      model = getattr(frame_level_models, model_name, None)()
      output = model.create_model(model_input, vocab_size, num_frames, l2_penalty=l2_penalty, **unused_params)["predictions"]
      outputs.append(tf.expand_dims(output, axis=2))
    num_models = len(outputs)
    model_outputs = tf.concat(outputs, axis=2)
#    linear_combination = tf.get_variable("combine", shape=[vocab_size,num_models],
#        dtype=tf.float32, initializer=tf.zeros_initializer(),
#        regularizer=slim.l2_regularizer(l2_penalty))
#    combination = tf.nn.softmax(linear_combination)
    combination = tf.fill(dims=[vocab_size,num_models], value=1.0/num_models)
    output_sum = tf.einsum("ijk,jk->ij", model_outputs, combination)
    return {"predictions": output_sum}

