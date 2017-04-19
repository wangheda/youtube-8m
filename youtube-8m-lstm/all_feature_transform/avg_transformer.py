import tensorflow as tf

class AvgTransformer:
  def transform(self, model_input_raw, num_frames, **unused_params):
    float_num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    feature_size = model_input_raw.get_shape().as_list()[2]
    denominators = tf.reshape(
        tf.tile(float_num_frames, [1, feature_size]), [-1, feature_size])
    avg_pooled = tf.reduce_sum(model_input_raw, axis=[1]) / denominators
    feature_dim = len(avg_pooled.get_shape()) - 1
    model_input = tf.nn.l2_normalize(avg_pooled, feature_dim)
    return model_input, num_frames
