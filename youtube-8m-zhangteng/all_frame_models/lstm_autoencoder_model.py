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

class LstmAutoencoderModel(models.BaseModel):

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

        cnn_input = tf.reshape(model_input,[-1,shape[2]])
        with tf.name_scope("autoencoder"):
            hidden_1 = slim.fully_connected(
                cnn_input,
                32,
                activation_fn=tf.nn.elu,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="hidden_1")
            hidden_1 = tf.reshape(tf.transpose(tf.reshape(hidden_1,[-1,shape[1],32]),[0,2,1]),[-1,shape[1]])
            hidden_2 = slim.fully_connected(
                hidden_1,
                32,
                activation_fn=tf.nn.elu,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="hidden_2")
            output_1 = slim.fully_connected(
                hidden_2,
                shape[1],
                activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="output_1")
            output_1 = tf.reshape(tf.transpose(tf.reshape(output_1,[-1,32,shape[1]]),[0,2,1]),[-1,32])
            output_2 = slim.fully_connected(
                output_1,
                shape[2],
                activation_fn=tf.nn.sigmoid,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="output_2")

        de_outputs = tf.reshape(output_2,[-1,shape[1],shape[2]])
        de_outputs = de_outputs*frames_bool
        video_input = tf.reshape(hidden_2,[-1,32*32])
        video_input = tf.nn.l2_normalize(video_input, dim=1)
        aggregated_model = getattr(video_level_models,"MoeCombineLayersModel")
        result = aggregated_model().create_model(model_input=video_input,vocab_size=vocab_size,
                                                 **unused_params)
        #result["bottleneck"] = state_c
        mse_loss = tf.square(de_outputs-model_input)
        mse_loss = tf.reduce_mean(tf.reduce_sum(mse_loss, 2))
        result["loss"] = mse_loss
        return result