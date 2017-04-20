
import tensorflow as tf
from tensorflow import flags
FLAGS = flags.FLAGS

class NoiseAugmenter:
  """This only works with frame data"""
  def augment(self, model_input_raw, num_frames, labels_batch, **unused_params):
    print "NoiseAugmenter", model_input_raw.shape, "noise =", FLAGS.input_noise_level
    noise_input = tf.random_normal(tf.shape(model_input_raw), mean=0.0, stddev=FLAGS.input_noise_level)
    model_input = model_input_raw + noise_input
    return model_input, labels_batch, num_frames
