
import tensorflow as tf

class DefaultTransformer:
  def transform(self, model_input_raw, num_frames, **unused_params):
    feature_dim = len(model_input_raw.get_shape()) - 1
    model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)
    return model_input, num_frames
