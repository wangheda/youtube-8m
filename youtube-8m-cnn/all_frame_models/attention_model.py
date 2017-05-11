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

class AttentionModel(models.BaseModel):

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
        frames_sum = tf.reduce_sum(tf.abs(model_input),axis=2)
        frames_true = tf.ones(tf.shape(frames_sum))
        frames_false = tf.zeros(tf.shape(frames_sum))
        frames_bool = tf.where(tf.greater(frames_sum, frames_false), frames_true, frames_false)

        num_extend = FLAGS.moe_num_extend

        shape = model_input.get_shape().as_list()
        denominators = tf.reshape(
            tf.tile(tf.cast(tf.expand_dims(num_frames, 1), tf.float32), [1, shape[2]]), [-1, shape[2]])
        avg_pooled = tf.reduce_sum(model_input,
                                   axis=[1]) / denominators
        avg_pooled = tf.tile(tf.reshape(avg_pooled,[-1,1,shape[2]]),[1,shape[1],1])

        attention_input = tf.reshape(tf.concat([model_input,avg_pooled],axis=2),[-1, shape[2]*2])

        with tf.variable_scope("Attention"):
            W = tf.Variable(tf.truncated_normal([shape[2]*2, num_extend], stddev=0.1), name="W")
            tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(W))
            b = tf.Variable(tf.constant(0.1, shape=[num_extend]), name="b")
            tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(b))
            output = tf.nn.softmax(tf.reshape(tf.nn.xw_plus_b(attention_input,W,b),[-1,shape[1],num_extend]),dim=1)
            output = output*tf.reshape(frames_bool,[-1,shape[1],1])
            atten = output/tf.reduce_sum(output, axis=1, keep_dims=True)

        state = tf.reduce_sum(tf.reshape(model_input,[-1,shape[1],1,shape[2]])*tf.reshape(atten,[-1,shape[1],num_extend,1]),axis=1)
        state = tf.reshape(state,[-1,shape[2]])

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=state,
            vocab_size=vocab_size,
            **unused_params)