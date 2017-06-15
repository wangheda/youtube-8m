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

class LstmAttentionModel(models.BaseModel):

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
        frames_bool = tf.where(tf.greater(frames_sum, frames_false), frames_true, frames_false)
        frames_bool = tf.reshape(frames_bool,[-1,shape[1]])

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
        state_c = tf.concat(map(lambda x: x.c, state), axis=1)
        state_h = tf.concat(map(lambda x: x.h, state), axis=1)
        state_all = tf.concat((state_c, state_h), axis=1)
        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        vocab_input = state_c
        attention_predictions = []
        class_size = FLAGS.class_size
        channels = vocab_size//class_size + 1

        hidden_outputs = tf.reshape(outputs,[-1,lstm_size])
        k = tf.get_variable("AttnW", [lstm_size,lstm_size], tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(k))
        hidden_features = tf.matmul(hidden_outputs, k)
        hidden_features = tf.reshape(hidden_features,[-1,shape[1],lstm_size])
        #v = tf.get_variable("AttnV", [lstm_size,1], tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        #tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(v))
        state_w = tf.get_variable("state_w", [lstm_size*2,lstm_size], tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(state_w))
        state_v = tf.get_variable("state_v" , [lstm_size], tf.float32,
                                  initializer=tf.constant_initializer(0.0))
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(state_v))
        def attention(query):
            """Put attention masks on hidden using hidden_features and query."""
            y = tf.reshape(query, [-1,lstm_size,1])
            # Attention mask is a softmax of v^T * tanh(...).
            s = tf.reshape(tf.einsum('aij,ajk->aik', hidden_features, y), [-1,shape[1]])
            a = tf.nn.softmax(s)
            rnn_attention = a*frames_bool
            rnn_attention = rnn_attention/(tf.reduce_sum(rnn_attention, axis=1, keep_dims=True)+1e-6)
            # Now calculate the attention-weighted vector d.
            d = tf.einsum('aij,ajk->aik', tf.reshape(rnn_attention, [-1, 1, shape[1]]), outputs)
            ds = tf.reshape(d, [-1, lstm_size])
            return ds

        for i in range(channels):
            if i<channels-1:
                sub_vocab_size = class_size
            else:
                sub_vocab_size = vocab_size-(channels-1)*class_size
            with tf.variable_scope("Attention"):
                with tf.variable_scope("moe-%s" % i):
                    attention_prediction = aggregated_model().create_model(model_input=state_c,vocab_size=sub_vocab_size,**unused_params)
                attention_input = attention_prediction["predictions"]
                if i==0:
                    attention_predictions = attention_input
                else:
                    attention_predictions = tf.concat((attention_predictions, attention_input),axis=1)
                state_input = tf.nn.xw_plus_b(state_all,state_w,state_v)
                attns = attention(state_input)
                linear_w = tf.get_variable("linear_w-%s" % i, [sub_vocab_size,class_size], tf.float32,
                                           initializer=tf.contrib.layers.xavier_initializer())
                tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(linear_w))
                linear_v = tf.get_variable("linear_v-%s" % i, [class_size], tf.float32,
                                           initializer=tf.constant_initializer(0.0))
                tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(linear_v))
                linear_input = tf.nn.relu(tf.nn.xw_plus_b(tf.stop_gradient(attention_input),linear_w,linear_v))
                if i==0:
                    reuse_flag = False
                else:
                    reuse_flag = True
                with tf.variable_scope("lstm",reuse=reuse_flag):
                    cell_output, state = stacked_lstm(tf.concat((linear_input,attns),axis=1), state)
                state_c = tf.concat(map(lambda x: x.c, state), axis=1)
                state_h = tf.concat(map(lambda x: x.h, state), axis=1)
                state_all = tf.concat((state_c,state_h),axis=1)

        probabilities_by_class = attention_predictions
        class_input = slim.fully_connected(
            probabilities_by_class,
            class_size,
            activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="class_inputs")
        vocab_input = tf.concat((vocab_input,class_input),axis=1)
        result = aggregated_model().create_model(model_input=vocab_input,vocab_size=vocab_size, **unused_params)
        result["predictions_class"] = probabilities_by_class

        return result