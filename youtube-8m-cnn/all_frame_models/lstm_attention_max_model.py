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

class LstmAttentionMaxModel(models.BaseModel):

    def create_model(self, model_input, vocab_size, num_frames,l2_penalty=1e-8, **unused_params):
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
        lstm_size = FLAGS.lstm_cells
        number_of_layers = FLAGS.lstm_layers
        num_extend = FLAGS.moe_num_extend
        shape = model_input.get_shape().as_list()
        frames_sum = tf.reduce_sum(tf.abs(model_input),axis=2)
        frames_true = tf.ones(tf.shape(frames_sum))
        frames_false = tf.zeros(tf.shape(frames_sum))
        frames_bool = tf.where(tf.greater(frames_sum, frames_false), frames_true, frames_false)
        frames_bool = tf.reshape(frames_bool,[-1,shape[1],1])

        ## Batch normalize the input
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)

        with tf.variable_scope("RNN"):
            outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                               sequence_length=num_frames,
                                               swap_memory=True,
                                               dtype=tf.float32)
        W1 = tf.Variable(tf.truncated_normal([lstm_size,num_extend], stddev=0.1), name="W1")
        b1 = tf.Variable(tf.constant(0.1, shape=[num_extend]), name="b1")
        W2 = tf.Variable(tf.truncated_normal([shape[2],num_extend], stddev=0.1), name="W2")
        b2 = tf.Variable(tf.constant(0.1, shape=[num_extend]), name="b2")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(W1))
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(b1))
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(W2))
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(b2))
        rnn_output = tf.reshape(outputs,[-1,lstm_size])
        rnn_input = tf.reshape(model_input,[-1,shape[2]])
        rnn_attention = tf.nn.softmax(tf.reshape(tf.nn.xw_plus_b(rnn_output,W1,b1)+tf.nn.xw_plus_b(rnn_input,W2,b2),[-1,shape[1],num_extend]),dim=1)*frames_bool
        rnn_attention = rnn_attention/tf.reduce_sum(rnn_attention, axis=1, keep_dims=True)
        rnn_attention = tf.transpose(rnn_attention,perm=[0,2,1])
        rnn_out = tf.einsum('aij,ajk->aik', rnn_attention, outputs)

        pooled = tf.reshape(rnn_out,[-1,lstm_size])

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=pooled,
            vocab_size=vocab_size,
            **unused_params)