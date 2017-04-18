
import tensorflow as tf
from tensorflow import flags
FLAGS = flags.FLAGS

class DefaultAugmenter:
  """This only works with frame data"""
  def augment(self, model_input_raw, num_frames, labels_batch, **unused_params):
    print "DefaultAugmenter"
    return model_input_raw, labels_batch, num_frames
