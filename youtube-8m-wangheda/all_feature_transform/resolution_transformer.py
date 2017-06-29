
import tensorflow as tf
from tensorflow import flags
FLAGS = flags.FLAGS

class ResolutionTransformer:
  def resolution(self, model_input_raw, num_frames):
    resolution = FLAGS.time_resolution

    frame_dim = len(model_input_raw.get_shape()) - 2
    feature_dim = len(model_input_raw.get_shape()) - 1

    max_frames = model_input_raw.get_shape().as_list()[frame_dim]
    num_features = model_input_raw.get_shape().as_list()[feature_dim]

    new_max_frames = max_frames / resolution
    cut_frames = new_max_frames * resolution

    model_input_raw = model_input_raw[:, :cut_frames, :]
    model_input_raw = tf.reshape(model_input_raw, shape=[-1,new_max_frames,resolution,num_features])
    model_input_raw = tf.reduce_mean(model_input_raw, axis=2)
    num_frames = num_frames / resolution
    return model_input_raw, num_frames

  def transform(self, model_input_raw, num_frames, **unused_params):
    model_input_raw, num_frames = self.resolution(model_input_raw, num_frames)
    feature_dim = len(model_input_raw.get_shape()) - 1
    model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)
    return model_input, num_frames
