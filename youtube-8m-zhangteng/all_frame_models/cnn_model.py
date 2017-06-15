import math

import models
import video_level_models
import tensorflow as tf
import model_utils as utils
import numpy as np

import tensorflow.contrib.slim as slim
from tensorflow import flags
import rnn_residual

FLAGS = flags.FLAGS

class CnnModel(models.BaseModel):
    # highway layer that borrowed from https://github.com/carpedm20/lstm-char-cnn-tensorflow
    def highway(self, input_1, input_2, size_1, size_2, l2_penalty=1e-8, layer_size=1):
        """Highway Network (cf. http://arxiv.org/abs/1505.00387).

        t = sigmoid(Wy + b)
        z = t * g(Wy + b) + (1 - t) * y
        where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
        """
        output = input_2

        for idx in range(layer_size):
            with tf.name_scope('output_lin_%d' % idx):
                W = tf.Variable(tf.truncated_normal([size_2,size_1], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[size_1]), name="b")
                tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(W))
                tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(b))
                output = tf.nn.relu(tf.nn.xw_plus_b(output,W,b))
            with tf.name_scope('transform_lin_%d' % idx):
                W = tf.Variable(tf.truncated_normal([size_1,size_1], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[size_1]), name="b")
                tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(W))
                tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(b))
                transform_gate = tf.sigmoid(tf.nn.xw_plus_b(input_1,W,b))
            carry_gate = tf.constant(1.0) - transform_gate

            output = transform_gate * output + carry_gate * input_1

        return output

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
        filter_sizes = [1, 2, 3]
        shape = model_input.get_shape().as_list()
        num_filters = [384,384,384]
        #num_filters = [200]
        pooled_outputs = []
        frames_sum = tf.reduce_sum(tf.abs(model_input),axis=2)
        frames_true = tf.ones(tf.shape(frames_sum))
        frames_false = tf.zeros(tf.shape(frames_sum))
        frames_bool = tf.where(tf.greater(frames_sum, frames_false), frames_true, frames_false)


        with tf.variable_scope("CNN"):

            num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
            denominators = tf.reshape(
                tf.tile(num_frames, [1, shape[2]]), [-1, shape[2]])
            avg_pooled = tf.reduce_sum(model_input,
                                       axis=[1]) / denominators

            cnn_input = tf.expand_dims(model_input, 3)

            for filter_size, num_filter in zip(filter_sizes, num_filters):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, shape[2], 1, num_filter]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
                    tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(W))
                    tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(b))
                    conv = tf.nn.conv2d(cnn_input,W,strides=[1, 1, 1, 1],padding="VALID",name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    h_shape = h.get_shape().as_list()
                    h_out = tf.reshape(h,[-1,h_shape[1],num_filter])
                    #frame_bool = frames_bool[:,0:filter_size*h_shape[1]:filter_size]
                    frame_bool = frames_bool[:,0:h_shape[1]]
                    frame_bool = tf.reshape(frame_bool/tf.reduce_sum(frame_bool,axis=1,keep_dims=True),[-1, h_shape[1], 1])
                    # Maxpooling over the outputs
                    pooled = tf.reduce_max(h_out*frame_bool, axis=1)
                    pooled_outputs.append(pooled)



            # Combine all the pooled features
            num_filters_total = sum(num_filters)
            h_pool = tf.concat(pooled_outputs,1)

            # Add highway
            with tf.name_scope("highway"):
                h_highway = self.highway(h_pool, h_pool, num_filters_total, num_filters_total, l2_penalty=l2_penalty)
            # Add dropout
            with tf.name_scope("dropout"):
                h_drop = tf.nn.dropout(h_highway, 0.5)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=h_drop,
            vocab_size=vocab_size,
            **unused_params)