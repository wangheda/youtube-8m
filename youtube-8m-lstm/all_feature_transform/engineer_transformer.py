
import tensorflow as tf
import numpy as np
from tensorflow import flags
FLAGS = flags.FLAGS

class EngineerTransformer:
  """feature transform by feature engineering"""
  def transform(self, model_input_raw, num_frames, **unused_params):
    feature_dim = len(model_input_raw.get_shape()) - 1
    engineer_types = map(lambda x: x.strip(), FLAGS.engineer_types.split(","))
    feature_list = []
    mask = self.mask(model_input_raw, num_frames)
    for etype in engineer_types:
      if etype == "avg":
        feature_list.append(self.avg(model_input_raw, num_frames, mask))
      elif etype == "std":
        feature_list.append(self.std(model_input_raw, num_frames, mask))
      elif etype == "diff":
        feature_list.append(self.diff(model_input_raw, num_frames, mask))
      else:
        feature_list.append(model_input_raw)
    model_input = tf.nn.l2_normalize(tf.concat(model_input_raw, axis=feature_dim), feature_dim)
    return model_input, num_frames

  def mask(self, model_input_raw, num_frames):
    max_frames = model_input_raw.get_shape().as_list()[1]
    mask_array = []
    for i in xrange(max_frames + 1):
      tmp = [0.0] * max_frames 
      for j in xrange(i):
        tmp[j] = 1.0
      mask_array.append(tmp)
    mask_array = np.array(mask_array)
    mask_init = tf.constant_initializer(mask_array)
    mask_emb = tf.get_variable("mask_emb", shape = [max_frames + 1, max_frames], 
            dtype = tf.float32, trainable = False, initializer = mask_init)
    mask = tf.nn.embedding_lookup(mask_emb, num_frames)
    return mask

  def avg(self, model_input_raw, num_frames, mask):
    max_frames = model_input_raw.get_shape().as_list()[1]
    num_frames_matrix = tf.maximum(tf.cast(
          tf.expand_dims(num_frames, axis=1), 
          dtype=tf.float32), 1.0)
    mean_matrix = mask / num_frames_matrix
    mean_input = tf.einsum("ijk,ij->ik", model_input_raw, mean_matrix)
    mean_input_tile = tf.tile(tf.expand_dims(mean_input, axis=1), multiples=[1,max_frames,1])
    return mean_input_tile

  def std(self, model_input_raw, num_frames, mask):
    mean_input = self.avg(model_input_raw, num_frames, mask)
    error = tf.einsum("ijk,ij->ijk", model_input_raw - mean_input, mask)
    return error 

  def diff(self, model_input_raw, num_frames, mask):
    max_frames = model_input_raw.get_shape().as_list()[1]
    shift_input1 = tf.pad(model_input_raw, paddings=[[0,0], [0,1], [0,0]])
    shift_input2 = tf.pad(model_input_raw, paddings=[[0,0], [1,0], [0,0]])
    diff_input = shift_input1 - shift_input2
    difference = tf.einsum("ijk,ij->ijk", diff_input[:,:max_frames,:], mask)
    return difference

