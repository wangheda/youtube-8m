
import tensorflow as tf
from tensorflow import flags
FLAGS = flags.FLAGS

class ClippingAugmenter:
  """This only works with frame data"""
  def augment(self, model_input_raw, num_frames, labels_batch, **unused_params):
    assert(FLAGS.frame_feature, 
           "AugmentationTransformer only works with frame feature")
    feature_dim = len(model_input_raw.get_shape()) - 1
    frame_dim = len(model_input_raw.get_shape()) - 2
    max_frame = model_input_raw.get_shape().as_list()[frame_dim]

    limit = tf.cast(tf.reduce_min(num_frames) / 4.0, tf.int32)
    offset = tf.random_uniform(shape=[], dtype=tf.int32) % limit
    input_trans1 = tf.pad(model_input_raw[:,offset:,:], paddings=[0,offset,0])
    num_frames_trans1 = num_frames - offset
    num_frames_trans1 = tf.cast(
                tf.random_uniform(shape=num_frames.shape, minval=0.75, maxval=1.0, 
                                  dtype=tf.float32) 
                * num_frames_trans1, tf.int32)
    model_input = tf.concat([model_input_raw, input_trans1], axis=0)
    labels_batch = tf.concat([labels_batch, labels_batch], axis=0)
    num_frames = tf.concat([num_frames, num_frames_trans1], axis=0)
    return model_input, labels_batch, num_frames_new
