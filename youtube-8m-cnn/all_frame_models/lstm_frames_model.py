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

class LstmFramesModel(models.BaseModel):

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
        shape = model_input.get_shape().as_list()

        frames_sum = tf.reduce_sum(tf.abs(model_input),axis=2)
        frames_true = tf.ones(tf.shape(frames_sum))
        frames_false = tf.zeros(tf.shape(frames_sum))
        frames_bool = tf.reshape(tf.where(tf.greater(frames_sum, frames_false), frames_true, frames_false),[-1,shape[1],1])

        ## Batch normalize the input
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=True)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=True)

        with tf.variable_scope("RNN"):
            outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                               sequence_length=num_frames,
                                               swap_memory=True,
                                               dtype=tf.float32)

        ## Batch normalize the input
        sigmoid_input = tf.reshape(outputs[:,0:shape[1]:FLAGS.stride_size,:],[-1,lstm_size])
        frames_bool = frames_bool[:,0:shape[1]:FLAGS.stride_size,:]
        probabilities_by_batch = slim.fully_connected(
            sigmoid_input,
            vocab_size,
            activation_fn=tf.nn.sigmoid,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="expert_activations")

        result = {}
        #result['prediction_frames'] = probabilities_by_batch
        probabilities_by_frame = tf.reshape(probabilities_by_batch,[-1,shape[1]//FLAGS.stride_size,vocab_size])*frames_bool
        probabilities_by_frame = tf.transpose(probabilities_by_frame,[0,2,1])
        probabilities_topk,_ = tf.nn.top_k(probabilities_by_frame, k=5)
        probabilities_by_frame = tf.transpose(probabilities_topk,[0,2,1])
        result['predictions'] = tf.reduce_mean(probabilities_by_frame,axis=1)
        return result