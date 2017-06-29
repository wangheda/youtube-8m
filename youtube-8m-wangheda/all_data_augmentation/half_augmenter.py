
import tensorflow as tf
from tensorflow import flags
FLAGS = flags.FLAGS

class HalfAugmenter:
  """This only works with frame data"""
  def augment(self, model_input_raw, num_frames, labels_batch, **unused_params):

    assert(FLAGS.frame_features, 
           "HalfAugmenter only works with frame feature")
    print "using HalfAugmeter"

    feature_dim = len(model_input_raw.get_shape()) - 1
    frame_dim = len(model_input_raw.get_shape()) - 2
    max_frame = model_input_raw.get_shape().as_list()[frame_dim]
    seg_length = max(int(max_frame / 2), 1)
    seg_num_frames = tf.maximum(num_frames / 2, 1)

    seg_inputs = []
    seg_frames = []
    seg_labels = []

    seg_inputs.append(model_input_raw)
    seg_frames.append(num_frames)
    seg_labels.append(labels_batch)

    for i in xrange(2):
      begin_frames = tf.reshape(seg_num_frames*i, [-1,1])
      frames_index = tf.reshape(tf.range(seg_length), [1,seg_length])
      frames_index = begin_frames + frames_index
      batch_size = tf.shape(model_input_raw)[0]
      batch_index = tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, seg_length])
      index = tf.stack([batch_index, tf.cast(frames_index,dtype=tf.int32)], 2)
      seg_input = tf.gather_nd(model_input_raw, index)
      seg_input = tf.pad(seg_input, paddings=[[0,0],[0, max_frame-seg_length],[0,0]])
      seg_input = seg_input * tf.expand_dims(tf.sequence_mask(seg_num_frames, maxlen=max_frame, dtype=tf.float32), axis=2)
      seg_inputs.append(seg_input)
      seg_frames.append(seg_num_frames)
      seg_labels.append(labels_batch)

    new_input_raw = tf.concat(seg_inputs, axis=0)
    new_num_frames = tf.concat(seg_frames, axis=0)
    new_labels_batch = tf.concat(seg_labels, axis=0)
    return new_input_raw, new_labels_batch, new_num_frames
