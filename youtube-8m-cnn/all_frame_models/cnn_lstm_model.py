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

class CnnLstmModel(models.BaseModel):

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
        num_frames = num_frames-filter_sizes[-1]+1
        shape = model_input.get_shape().as_list()
        slice = shape[1]-filter_sizes[-1]+1
        num_filters = [384,384,384]
        pooled_outputs = []

        frames_sum = tf.reduce_sum(tf.abs(model_input),axis=2)
        frames_true = tf.ones(tf.shape(frames_sum))
        frames_false = tf.zeros(tf.shape(frames_sum))
        frames_bool = tf.where(tf.greater(frames_sum, frames_false), frames_true, frames_false)
        frames_bool = tf.reshape(frames_bool,[-1,shape[1],1])
        frames_bool = frames_bool/tf.reduce_sum(frames_bool,axis=1,keep_dims=True)

        with tf.variable_scope("CNN"):

            cnn_input = tf.expand_dims(model_input, 3)

            for filter_size, num_filter in zip(filter_sizes, num_filters):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    #paddings = tf.constant([[0,0],[filter_size//2,filter_size-filter_size//2-1],[0,0]])
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
                    #h_out = tf.pad(h_out,paddings)
                    h_out = h_out[:,0:slice,:]
                    pooled_outputs.append(h_out)



            # Combine all the pooled features
            h_pool = tf.concat(pooled_outputs,2)

        lstm_size = FLAGS.lstm_cells
        number_of_layers = FLAGS.lstm_layers

        ## Batch normalize the input
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)

        with tf.variable_scope("RNN"):
            outputs, state = tf.nn.dynamic_rnn(stacked_lstm, h_pool,
                                               sequence_length=num_frames,
                                               swap_memory=True,
                                               dtype=tf.float32)
        output = tf.reduce_sum(outputs*frames_bool[:,0:slice,:],axis=1)
        #output = tf.reduce_max(outputs,axis=1)
        pooled = tf.concat((output,state),axis=1)
        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=pooled,
            vocab_size=vocab_size,
            **unused_params)