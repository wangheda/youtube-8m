import sys
import models
import model_utils
import math
import numpy as np
import video_level_models
import tensorflow as tf
import utils
import tensorflow.contrib.slim as slim
from tensorflow import flags
FLAGS = flags.FLAGS

class CnnKmaxModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, l2_penalty=1e-8, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    # Create a convolution + maxpool layer for each filter size
    filter_sizes = map(int, FLAGS.cnn_filter_sizes.split(","))
    filter_nums = map(int, FLAGS.cnn_filter_nums.split(","))
    pooling_k = FLAGS.cnn_pooling_k

    assert len(filter_sizes) == len(filter_nums), \
      "length of filter_sizes (={}) != length of filter_nums (={})".format( \
      len(filter_sizes), len(filter_nums))

    batch_size, max_frames, num_features = model_input.get_shape().as_list()

    with tf.variable_scope("CNN"):

      # set channel dimension to 1
      # cnn_input.shape = [batch, in_height, in_width, in_channels]
      cnn_input = tf.expand_dims(model_input, axis = 3)
            
      cnn_output = []
      for filter_size, filter_num in zip(filter_sizes, filter_nums):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
          # Convolution Layer
          # filter.shape = [filter_height, filter_width, in_channels, out_channels]
          filter_shape = [filter_size, num_features, 1, filter_num]

          W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
          b = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="b")

          tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(W))
          tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(b))

          # conv.shape = [batch, -1, 1, out_channels]
          conv = tf.nn.conv2d(cnn_input, W, strides=[1, 1, 1, 1], padding="VALID",name="conv")
          # add bias
          conv = tf.nn.bias_add(conv, b)
          if pooling_k == 1:
            _, conv_len, _, _= conv.shape.as_list()
            conv_kmax = tf.reduce_max(conv, axis = 1)
            conv_flat = tf.reshape(conv_kmax, [-1, filter_num])
          else:
            conv_kmax = tf.transpose(tf.squeeze(conv, axis = 2), perm = [0, 2, 1])
            conv_kmax, _ = tf.nn.top_k(conv_kmax, k = pooling_k, sorted = True)
            conv_flat = tf.reshape(conv_kmax, [-1, filter_num * pooling_k])
          cnn_output.append(conv_flat)
      cnn_output = tf.concat(cnn_output, axis = 1)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=conv_flat,
        original_input=model_input,
        vocab_size=vocab_size,
        **unused_params)

