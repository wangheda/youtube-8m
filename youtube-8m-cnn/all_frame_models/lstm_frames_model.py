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

    def sub_moe(self,
                model_input,
                vocab_size,
                num_mixtures=None,
                l2_penalty=1e-8,
                **unused_params):
        """Creates a Mixture of (Logistic) Experts model.

         The model consists of a per-class softmax distribution over a
         configurable number of logistic classifiers. One of the classifiers in the
         mixture is not trained, and always predicts 0.

        Args:
          model_input: 'batch_size' x 'num_features' matrix of input features.
          vocab_size: The number of classes in the dataset.
          num_mixtures: The number of mixtures (excluding a dummy 'expert' that
            always predicts the non-existence of an entity).
          l2_penalty: How much to penalize the squared magnitudes of parameter
            values.
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes.
        """
        num_mixtures = num_mixtures or FLAGS.moe_num_mixtures
        num_extends = FLAGS.moe_layers

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures


        final_probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)

        final_probabilities = tf.reduce_mean(tf.reshape(final_probabilities_by_class_and_batch,
                                                        [-1, num_extends, vocab_size]),axis=1)

        return {"predictions": final_probabilities}

    def calculate_loss(self, predictions, labels, **unused_params):
        with tf.name_scope("loss_xent"):
            epsilon = 10e-6
            float_labels = tf.cast(labels, tf.float32)
            cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + (
                1 - float_labels) * tf.log(1 - predictions + epsilon)

        return tf.reduce_sum(cross_entropy_loss, 2)


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

        unit_layer_1 = FLAGS.stride_size
        unit_layer_2 = shape[1]//FLAGS.stride_size
        model_input = model_input[:,:unit_layer_1*unit_layer_2,:]

        model_input_1 = tf.reshape(model_input,[-1,unit_layer_1,shape[2]])
        ## Batch normalize the input
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=True)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=True)

        with tf.variable_scope("RNN-1"):
            outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input_1,
                                               sequence_length=None,
                                               swap_memory=True,
                                               dtype=tf.float32)

        ## Batch normalize the input
        sigmoid_input = tf.concat(map(lambda x: x.c, state), axis=1)
        frames_bool = frames_bool[:,0:shape[1]:FLAGS.stride_size,:]
        probabilities_by_batch = slim.fully_connected(
            sigmoid_input,
            vocab_size,
            activation_fn=tf.nn.sigmoid,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="expert_activations")

        probabilities_by_frame = tf.reshape(probabilities_by_batch,[-1,shape[1]//FLAGS.stride_size,vocab_size])*frames_bool
        probabilities_transpose = tf.transpose(probabilities_by_frame,[0,2,1])
        probabilities_topk, _ = tf.nn.top_k(probabilities_transpose, k=FLAGS.moe_layers)
        probabilities_topk = tf.reshape(tf.transpose(probabilities_topk, [0,2,1]),[-1,vocab_size*FLAGS.moe_layers])

        importance_by_frame = self.calculate_loss(probabilities_by_frame,tf.reduce_mean(
            tf.reshape(probabilities_topk,[-1,FLAGS.moe_layers,vocab_size]),axis=1,keep_dims=True))
        _, index_topk = tf.nn.top_k(importance_by_frame, k=FLAGS.moe_layers)

        batch_size = tf.shape(model_input)[0]
        batch_index = tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, FLAGS.moe_layers])
        index = tf.reshape(batch_index*(shape[1]//FLAGS.stride_size) + index_topk,[-1])
        moe_input = tf.reshape(tf.gather(sigmoid_input, index),[-1,FLAGS.moe_layers,lstm_size*number_of_layers])
        result = self.sub_moe(moe_input,vocab_size)
        result["predictions_class"] = probabilities_topk
        return result