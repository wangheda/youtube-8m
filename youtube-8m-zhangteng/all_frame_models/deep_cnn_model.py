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

class DeepCnnModel(models.BaseModel):

    def highway(self, input_1, input_2, size_1, size_2, l2_penalty=1e-8, layer_size=1):
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

    def conv_block(self, input, out_size, layer, kernalsize=3, l2_penalty=1e-8, shortcut=False):
        in_shape = input.get_shape().as_list()
        if layer>0:
            filter_shape = [kernalsize, 1, in_shape[3], out_size]
        else:
            filter_shape = [kernalsize, in_shape[2], 1, out_size]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W-%s" % layer)
        b = tf.Variable(tf.constant(0.1, shape=[out_size]), name="b-%s" % layer)
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(W))
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(b))
        if layer>0:
            conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding="SAME", name="conv-%s" % layer)
        else:
            conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding="VALID", name="conv-%s" % layer)
        if shortcut:
            shortshape = [1,1,in_shape[3], out_size]
            Ws = tf.Variable(tf.truncated_normal(shortshape, stddev=0.05), name="Ws-%s" % layer)
            tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Ws))
            conv = conv + tf.nn.conv2d(input, Ws, strides=[1, 1, 1, 1], padding="SAME", name="conv-shortcut-%s" % layer)
        h = tf.nn.bias_add(conv, b)
        h2 = tf.nn.relu(tf.contrib.layers.batch_norm(h, center=True, scale=True, epsilon=1e-5, decay=0.9), name="relu-%s" % layer)

        return h2

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
        kernal_size=3
        pool_size=2
        num_extend = FLAGS.moe_num_extend
        num_filters = 1024
        layers = 4


        with tf.variable_scope("CNN"):

            cnn_input = tf.expand_dims(model_input, 3)
            cnn_output = []
            for i in range(layers):
                with tf.name_scope("conv-maxpool-%s" % i):
                    # Convolution Layer
                    cnn_input = self.conv_block(cnn_input,l2_penalty=l2_penalty, out_size=num_filters, layer=i, kernalsize=kernal_size)
                    if i<layers-1:
                        cnn_input = tf.nn.max_pool(cnn_input,ksize=[1, pool_size, 1, 1],strides=[1, pool_size, 1, 1],padding='VALID',name="pool-%s" % i)
                    else:
                        cnn_output = tf.transpose(cnn_input, perm=[0,2,3,1])
                        cnn_output, indexes = tf.nn.top_k(cnn_output, k=num_extend)
                        cnn_output = tf.transpose(cnn_output, perm=[0,3,1,2])

            pooled = tf.reshape(cnn_output,[-1,num_filters])

            """
            # Add highway
            with tf.name_scope("highway"):
                h_highway = self.highway(pooled, pooled, num_filters, num_filters, l2_penalty=l2_penalty)
            # Add dropout
            with tf.name_scope("dropout"):
                h_drop = tf.nn.dropout(h_highway, 0.5)"""

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=pooled,
            vocab_size=vocab_size,
            **unused_params)