# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains a collection of models which operate on variable-length sequences.
"""
import math

import models
import video_level_models
import tensorflow as tf
import model_utils as utils
import numpy as np

import tensorflow.contrib.slim as slim
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
from tensorflow import flags
import rnn_residual

FLAGS = flags.FLAGS
flags.DEFINE_integer("iterations", 30,
                     "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")
flags.DEFINE_integer("dbof_cluster_size", 8192,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 1024,
                     "Number of units in the DBoF hidden layer.")
flags.DEFINE_string("dbof_pooling_method", "max",
                    "The pooling method used in the DBoF cluster layer. "
                    "Choices are 'average' and 'max'.")
flags.DEFINE_string("video_level_classifier_model", "MoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")
flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_length", 10, "Number of LSTM length, only used in LstmLayerModel.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")
flags.DEFINE_integer("lstm_interval", 3, "Number of LSTM residual intervals, only used in LstmResidualModel.")
flags.DEFINE_bool("train", True,
                    "Whether the process is training procedure."
                    "used for batch normalization and LstmRandomModel and LstmNioseModel.")

flags.DEFINE_integer("cnn_cells", 256, "Number of CNN cells.")

class FrameLevelLogisticModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a logistic classifier over the average of the
    frame-level features.

    This class is intended to be an example for implementors of frame level
    models. If you want to train a model over averaged features it is more
    efficient to average them beforehand rather than on the fly.

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
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    feature_size = model_input.get_shape().as_list()[2]
    max_frames = model_input.get_shape().as_list()[1]


    denominators = tf.reshape(
        tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
    avg_pooled = tf.reduce_sum(model_input,
                               axis=[1]) / denominators
    output = slim.fully_connected(
        avg_pooled, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(1e-8))
    return {"predictions": output}

class DbofModel(models.BaseModel):
  """Creates a Deep Bag of Frames model.

  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.

  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.

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

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    cluster_weights = tf.Variable(tf.random_normal(
        [feature_size, cluster_size],
        stddev=1 / math.sqrt(feature_size)))
    tf.summary.histogram("cluster_weights", cluster_weights)
    activation = tf.matmul(reshaped_input, cluster_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="cluster_bn")
    else:
      cluster_biases = tf.Variable(
          tf.random_normal(
              [cluster_size], stddev=1 / math.sqrt(feature_size)))
      tf.summary.histogram("cluster_biases", cluster_biases)
      activation += cluster_biases
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("cluster_output", activation)

    activation = tf.reshape(activation, [-1, max_frames, cluster_size])
    activation = utils.FramePooling(activation, FLAGS.dbof_pooling_method)

    hidden1_weights = tf.Variable(tf.random_normal(
        [cluster_size, hidden1_size],
        stddev=1 / math.sqrt(cluster_size)))
    tf.summary.histogram("hidden1_weights", hidden1_weights)
    activation = tf.matmul(activation, hidden1_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")
    else:
      hidden1_biases = tf.Variable(
          tf.random_normal(
              [hidden1_size], stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("hidden1_output", activation)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        **unused_params)

class Dbof3mModel(models.BaseModel):
    """Creates a Deep Bag of Frames model.

    The model projects the features for each frame into a higher dimensional
    'clustering' space, pools across frames in that space, and then
    uses a configurable video-level model to classify the now aggregated features.

    The model will randomly sample either frames or sequences of frames during
    training to speed up convergence.

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
    def sub_moe(self,
                model_input,
                vocab_size,
                num_mixtures = None,
                l2_penalty=1e-8,
                scopename="",
                **unused_params):

        num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates"+scopename)
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts"+scopename)

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures


        final_probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)

        final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                         [-1, vocab_size])
        return model_input, final_probabilities

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_frames,
                     **unused_params):

        shape = model_input.get_shape().as_list()
        frames_sum = tf.reduce_sum(tf.abs(model_input),axis=2)
        frames_true = tf.ones(tf.shape(frames_sum))
        frames_false = tf.zeros(tf.shape(frames_sum))
        frames_bool = tf.reshape(tf.where(tf.greater(frames_sum, frames_false), frames_true, frames_false),[-1,shape[1],1])

        activation_1 = tf.reduce_max(model_input, axis=1)
        activation_2 = tf.reduce_sum(model_input*frames_bool, axis=1)/(tf.reduce_sum(frames_bool, axis=1)+1e-6)
        activation_3 = tf.reduce_min(model_input, axis=1)

        model_input_1, final_probilities_1 = self.sub_moe(activation_1,vocab_size,scopename="_max")
        model_input_2, final_probilities_2 = self.sub_moe(activation_2,vocab_size,scopename="_mean")
        model_input_3, final_probilities_3 = self.sub_moe(activation_3,vocab_size,scopename="_min")
        final_probilities = tf.stack((final_probilities_1,final_probilities_2,final_probilities_3),axis=1)
        weight2d = tf.get_variable("ensemble_weight2d",
                                   shape=[shape[2], 3, vocab_size],
                                   regularizer=slim.l2_regularizer(1.0e-8))
        activations = tf.stack((model_input_1, model_input_2, model_input_3), axis=2)
        weight = tf.nn.softmax(tf.einsum("aij,ijk->ajk", activations, weight2d), dim=1)
        result = {}
        result["prediction_frames"] = tf.reshape(final_probilities,[-1,vocab_size])
        result["predictions"] = tf.reduce_sum(final_probilities*weight,axis=1)
        return result

class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum=0.99, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        with tf.variable_scope(self.name) as scope:
            self.beta = tf.get_variable("beta", shape[1:],
                                        initializer=tf.constant_initializer(0.))
            self.gamma = tf.get_variable("gamma", shape[1:],
                                         initializer=tf.random_normal_initializer(1.,0.02))
            self.mean = tf.get_variable("mean", shape[1:],
                                        initializer=tf.constant_initializer(0.),trainable=False)
            self.variance = tf.get_variable("variance",shape[1:],
                                            initializer=tf.constant_initializer(1.),trainable=False)
            if train:
                batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')

                self.mean.assign(batch_mean)
                self.variance.assign(batch_var)
                ema_apply_op = self.ema.apply([self.mean, self.variance])
                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
            else:
                mean, var = self.ema.average(self.mean), self.ema.average(self.variance)

            normed = tf.nn.batch_normalization(x, mean, var, self.beta, self.gamma, self.epsilon)

        return normed

class LstmVisionModel(models.BaseModel):
    """Logistic model with L2 regularization."""

    def create_model(self, model_input, vocab_size, num_frames, l2_penalty=1e-8, **unused_params):
        """Creates a matrix regression model.

        Args:
          model_input: 'batch' x 'num_features' x 'num_methods' matrix of input features.
          vocab_size: The number of classes in the dataset.

        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes."""

        lstm_size = FLAGS.lstm_cells
        max_frames = model_input.get_shape().as_list()[1]
        emb_dim = model_input.get_shape().as_list()[2]
        hidden_dim = lstm_size

        with tf.variable_scope('lstm_forward'):
            g_recurrent_unit_forward = self.create_recurrent_unit(emb_dim,hidden_dim,l2_penalty)

        h0 = tf.zeros([tf.shape(model_input)[0], hidden_dim])
        h0 = tf.stack([h0, h0])

        outputs_gate = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=max_frames,
            dynamic_size=False, infer_shape=True)
        outputs_state = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=max_frames,
            dynamic_size=False, infer_shape=True)

        def _pretrain_forward(i,h_tm1,g_predictions,s_predictions):
            x_t = model_input[:,i,:]
            gate, h_t = g_recurrent_unit_forward(x_t, h_tm1)
            hidden_state, c_prev = tf.unstack(h_t)
            g_predictions = g_predictions.write(i,gate)
            s_predictions = s_predictions.write(i,c_prev)
            return i + 1, h_t, g_predictions, s_predictions

        _, _, gate_outputs, state_outputs = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3: i < max_frames,
            body=_pretrain_forward,
            loop_vars=(tf.constant(0, dtype=tf.int32),h0,outputs_gate,outputs_state))

        gate_outputs = gate_outputs.stack()
        state_outputs = state_outputs.stack()
        batch_size = tf.shape(model_input)[0]
        index_1 = tf.range(0, batch_size) * max_frames + (num_frames - 1)
        gate_outputs = tf.transpose(gate_outputs, [1, 0, 2])
        gate_outputs = tf.gather(tf.reshape(gate_outputs, [-1, hidden_dim]), index_1)
        state_outputs = tf.transpose(state_outputs, [1, 0, 2])
        state_outputs = tf.gather(tf.reshape(state_outputs, [-1, hidden_dim]), index_1)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=state_outputs,
            vocab_size=vocab_size,
            **unused_params)

    def create_recurrent_unit(self,emb_dim,hidden_dim,l2_penalty):
        # Weights and Bias for input and hidden tensor
        Wi = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wi")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wi))
        Ui = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Ui")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Ui))
        bi = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bi")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bi))

        Wf = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wf))
        Uf = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Uf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uf))
        bf = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bf))

        Wog = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wog))
        Uog = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Uog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uog))
        bog = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bog))

        Wc = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wc))
        Uc = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Uc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uc))
        bc = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bc))

        bn0 = batch_norm(name='bn0')
        bn1 = batch_norm(name='bn1')
        bn2 = batch_norm(name='bn2')
        bn3 = batch_norm(name='bn3')
        gn0 = batch_norm(name='gn0')
        gn1 = batch_norm(name='gn1')
        gn2 = batch_norm(name='gn2')
        gn3 = batch_norm(name='gn3')

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)
            # Input Gate
            i = tf.sigmoid(
                bn0(tf.matmul(x, Wi), train=FLAGS.train) +
                gn0(tf.matmul(previous_hidden_state, Ui), train=FLAGS.train) + bi
            )
            # Forget Gate
            f = tf.sigmoid(
                bn1(tf.matmul(x, Wf), train=FLAGS.train) +
                gn1(tf.matmul(previous_hidden_state, Uf), train=FLAGS.train) + bf
            )
            # Output Gate
            o = tf.sigmoid(
                bn2(tf.matmul(x, Wog), train=FLAGS.train) +
                gn2(tf.matmul(previous_hidden_state, Uog), train=FLAGS.train) + bog
            )
            # New Memory Cell
            c_ = tf.nn.tanh(
                bn3(tf.matmul(x, Wc), train=FLAGS.train) +
                gn3(tf.matmul(previous_hidden_state, Uc), train=FLAGS.train) + bc
            )
            # Final Memory cell
            c = f * c_prev + i * c_
            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)
            return f, tf.stack([current_hidden_state, c])
        return unit

class LstmGluModel(models.BaseModel):
    """Logistic model with L2 regularization."""

    def create_model(self, model_input, vocab_size, num_frames, l2_penalty=1e-8, **unused_params):
        """Creates a matrix regression model.

        Args:
          model_input: 'batch' x 'num_features' x 'num_methods' matrix of input features.
          vocab_size: The number of classes in the dataset.

        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes."""

        lstm_size = FLAGS.lstm_cells
        max_frames = model_input.get_shape().as_list()[1]
        emb_dim = model_input.get_shape().as_list()[2]
        hidden_dim = lstm_size

        with tf.variable_scope('lstm_forward'):
            g_recurrent_unit_forward = self.create_recurrent_unit(emb_dim,hidden_dim,l2_penalty)

        h0 = tf.zeros([tf.shape(model_input)[0], hidden_dim])
        h1 = tf.zeros([tf.shape(model_input)[0], emb_dim])
        h0 = tf.stack([h0, h0])
        h1 = tf.stack([h1, h1])

        outputs_gate = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=max_frames,
            dynamic_size=False, infer_shape=True)
        outputs_state = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=max_frames,
            dynamic_size=False, infer_shape=True)

        def _pretrain_forward(i, h_tm0, h_tm1, g_predictions, s_predictions):
            x_t = model_input[:,i,:]
            gate, h_t0, h_t1 = g_recurrent_unit_forward(x_t, h_tm0, h_tm1)
            hidden_state, c_prev = tf.unstack(h_t1)
            g_predictions = g_predictions.write(i,gate)
            s_predictions = s_predictions.write(i,c_prev)
            return i + 1, h_t0, h_t1, g_predictions, s_predictions

        _, _, _, gate_outputs, state_outputs = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < max_frames,
            body=_pretrain_forward,
            loop_vars=(tf.constant(0, dtype=tf.int32), h1, h0,outputs_gate,outputs_state))

        gate_outputs = gate_outputs.stack()
        state_outputs = state_outputs.stack()
        batch_size = tf.shape(model_input)[0]
        index_1 = tf.range(0, batch_size) * max_frames + (num_frames - 1)
        gate_outputs = tf.transpose(gate_outputs, [1, 0, 2])
        gate_outputs = tf.gather(tf.reshape(gate_outputs, [-1, hidden_dim]), index_1)
        state_outputs = tf.transpose(state_outputs, [1, 0, 2])
        state_outputs = tf.gather(tf.reshape(state_outputs, [-1, hidden_dim]), index_1)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=state_outputs,
            vocab_size=vocab_size,
            **unused_params)

    def create_recurrent_unit(self,emb_dim,hidden_dim,l2_penalty):
        # Weights and Bias for input and hidden tensor
        Wi = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wi")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wi))
        Ui = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Ui")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Ui))
        Vi = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Vi")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Vi))
        bi = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bi")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bi))

        Wf = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wf))
        Vf = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Vf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Vf))
        Uf = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Uf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uf))
        bf = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bf))

        Wog = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wog))
        Vog = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Vog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Vog))
        Uog = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Uog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uog))
        bog = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bog))

        Wc = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wc))
        Vc = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Vc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Vc))
        Uc = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Uc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uc))
        bc = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bc))

        Wix = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Wix")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wix))
        Vix = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Vix")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Vix))
        Uix = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=0.1), name="Uix")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uix))
        bix = tf.Variable(tf.constant(0.1, shape=[1]), name="bix")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bix))

        Wfx = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Wfx")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wfx))
        Vfx = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Vfx")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Vfx))
        Ufx = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=0.1), name="Ufx")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Ufx))
        bfx = tf.Variable(tf.constant(0.1, shape=[1]), name="bfx")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bfx))



        def unit(x, hidden_memory_tm0, hidden_memory_tm1):
            previous_hidden_x, x_prev = tf.unstack(hidden_memory_tm0)
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)
            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, Wi) + tf.matmul(previous_hidden_x, Vi) +
                tf.matmul(previous_hidden_state, Ui) + bi
            )
            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, Wf) + tf.matmul(previous_hidden_x, Vf) +
                tf.matmul(previous_hidden_state, Uf) + bf
            )
            # Input Gate
            ix = tf.sigmoid(
                tf.matmul(x, Wix) + tf.matmul(previous_hidden_x, Vix) +
                tf.matmul(previous_hidden_state, Uix) + bix
            )
            # Forget Gate
            fx = tf.sigmoid(
                tf.matmul(x, Wfx) + tf.matmul(previous_hidden_x, Vfx) +
                tf.matmul(previous_hidden_state, Ufx) + bfx
            )
            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, Wog) + tf.matmul(previous_hidden_x, Vog) +
                tf.matmul(previous_hidden_state, Uog) + bog
            )
            # New Memory Cell
            c_ = tf.tanh(
                tf.matmul(x, Wc) + tf.matmul(previous_hidden_x, Vc) +
                tf.matmul(previous_hidden_state, Uc) + bc
                )

            # Final Memory cell
            c = f * c_prev + i * c_
            current_x = fx * x_prev + ix * x
            current_hidden_x = tf.nn.l2_normalize(current_x, dim=1)
            # Current Hidden state
            current_hidden_state = o * tf.tanh(c)
            return f, tf.stack([current_hidden_x, current_x]), tf.stack([current_hidden_state, c])
        return unit

class LstmGlu2Model(models.BaseModel):
    """Logistic model with L2 regularization."""

    def create_model(self, model_input, vocab_size, num_frames, l2_penalty=1e-8, **unused_params):
        """Creates a matrix regression model.

        Args:
          model_input: 'batch' x 'num_features' x 'num_methods' matrix of input features.
          vocab_size: The number of classes in the dataset.

        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes."""

        lstm_size = FLAGS.lstm_cells
        max_frames = model_input.get_shape().as_list()[1]
        emb_dim = model_input.get_shape().as_list()[2]
        hidden_dim = lstm_size

        with tf.variable_scope('lstm_forward'):
            g_recurrent_unit_forward = self.create_recurrent_unit(emb_dim,hidden_dim,l2_penalty)

        h0 = tf.zeros([tf.shape(model_input)[0], hidden_dim])
        h1 = tf.zeros([tf.shape(model_input)[0], emb_dim])
        h0 = tf.stack([h0, h0])
        h1 = tf.stack([h1, h1])

        outputs_gate = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=max_frames,
            dynamic_size=False, infer_shape=True)
        outputs_state = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=max_frames,
            dynamic_size=False, infer_shape=True)

        def _pretrain_forward(i, h_tm0, h_tm1, g_predictions, s_predictions):
            x_t = model_input[:,i,:]
            gate, h_t0, h_t1 = g_recurrent_unit_forward(x_t, h_tm0, h_tm1)
            hidden_state, c_prev = tf.unstack(h_t1)
            g_predictions = g_predictions.write(i,gate)
            s_predictions = s_predictions.write(i,c_prev)
            return i + 1, h_t0, h_t1, g_predictions, s_predictions

        _, _, _, gate_outputs, state_outputs = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < max_frames,
            body=_pretrain_forward,
            loop_vars=(tf.constant(0, dtype=tf.int32), h1, h0,outputs_gate,outputs_state))

        gate_outputs = gate_outputs.stack()
        state_outputs = state_outputs.stack()
        batch_size = tf.shape(model_input)[0]
        index_1 = tf.range(0, batch_size) * max_frames + (num_frames - 1)
        gate_outputs = tf.transpose(gate_outputs, [1, 0, 2])
        gate_outputs = tf.gather(tf.reshape(gate_outputs, [-1, hidden_dim]), index_1)
        state_outputs = tf.transpose(state_outputs, [1, 0, 2])
        state_outputs = tf.gather(tf.reshape(state_outputs, [-1, hidden_dim]), index_1)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=state_outputs,
            vocab_size=vocab_size,
            **unused_params)

    def create_recurrent_unit(self,emb_dim,hidden_dim,l2_penalty):
        # Weights and Bias for input and hidden tensor
        Wi = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Wi")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wi))
        Ui = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=0.1), name="Ui")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Ui))
        Vi = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Vi")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Vi))
        bi = tf.Variable(tf.constant(0.1, shape=[1]), name="bi")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bi))

        Wf = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Wf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wf))
        Vf = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Vf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Vf))
        Uf = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=0.1), name="Uf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uf))
        bf = tf.Variable(tf.constant(1.0, shape=[1]), name="bf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bf))

        Wog = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wog))
        Vog = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Vog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Vog))
        Uog = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Uog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uog))
        bog = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bog))

        Wc = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wc))
        Vc = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Vc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Vc))
        Uc = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Uc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uc))
        bc = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bc))

        Wix = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Wix")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wix))
        Vix = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Vix")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Vix))
        Uix = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=0.1), name="Uix")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uix))
        bix = tf.Variable(tf.constant(0.1, shape=[1]), name="bix")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bix))

        Wfx = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Wfx")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wfx))
        Vfx = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Vfx")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Vfx))
        Ufx = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=0.1), name="Ufx")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Ufx))
        bfx = tf.Variable(tf.constant(1.0, shape=[1]), name="bfx")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bfx))

        def unit(x, hidden_memory_tm0, hidden_memory_tm1):
            previous_hidden_x, x_prev = tf.unstack(hidden_memory_tm0)
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)
            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, Wi) + tf.matmul(previous_hidden_x, Vi) +
                tf.matmul(previous_hidden_state, Ui) + bi
            )
            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, Wf) + tf.matmul(previous_hidden_x, Vf) +
                tf.matmul(previous_hidden_state, Uf) + bf
            )
            # Input Gate
            ix = tf.sigmoid(
                tf.matmul(x, Wix) + tf.matmul(previous_hidden_x, Vix) +
                tf.matmul(previous_hidden_state, Uix) + bix
            )
            # Forget Gate
            fx = tf.sigmoid(
                tf.matmul(x, Wfx) + tf.matmul(previous_hidden_x, Vfx) +
                tf.matmul(previous_hidden_state, Ufx) + bfx
            )
            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, Wog) + tf.matmul(previous_hidden_x, Vog) +
                tf.matmul(previous_hidden_state, Uog) + bog
            )
            # New Memory Cell
            c_ = tf.tanh(
                tf.matmul(x, Wc) + tf.matmul(previous_hidden_x, Vc) +
                tf.matmul(previous_hidden_state, Uc) + bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_
            current_x = fx * x_prev + ix * x
            current_hidden_x = tf.nn.l2_normalize(current_x, dim=1)
            # Current Hidden state
            current_hidden_state = o * tf.tanh(c)
            return f, tf.stack([current_hidden_x, current_x]), tf.stack([current_hidden_state, c])
        return unit

class LstmGlu2MultilayerModel(models.BaseModel):
    """Logistic model with L2 regularization."""

    def create_model(self, model_input, vocab_size, num_frames, l2_penalty=1e-8, **unused_params):
        """Creates a matrix regression model.

        Args:
          model_input: 'batch' x 'num_features' x 'num_methods' matrix of input features.
          vocab_size: The number of classes in the dataset.

        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes."""

        lstm_size = FLAGS.lstm_cells
        number_of_layers = FLAGS.lstm_layers

        hidden_outputs = model_input
        state_outputs = []
        for i in range(number_of_layers):
            state_output, hidden_outputs = self.rnn_gate(hidden_outputs, lstm_size, num_frames, sub_scope="lstm_lsyer%d" % i)
            state_outputs.append(state_output)

        state_outputs = tf.concat(state_outputs,axis=1)
        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=state_outputs,
            vocab_size=vocab_size,
            **unused_params)

    def rnn_gate(self, model_input, lstm_size, num_frames, l2_penalty=1e-8, sub_scope="", **unused_params):
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
        ## Batch normalize the input

        max_frames = model_input.get_shape().as_list()[1]
        emb_dim = model_input.get_shape().as_list()[2]
        hidden_dim = lstm_size

        with tf.variable_scope(sub_scope+'lstm_forward'):
            g_recurrent_unit_forward = LstmGlu2Model().create_recurrent_unit(emb_dim,hidden_dim,l2_penalty)

        h0 = tf.zeros([tf.shape(model_input)[0], hidden_dim])
        h1 = tf.zeros([tf.shape(model_input)[0], emb_dim])
        h0 = tf.stack([h0, h0])
        h1 = tf.stack([h1, h1])

        outputs_hidden = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=max_frames,
            dynamic_size=False, infer_shape=True)
        outputs_state = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=max_frames,
            dynamic_size=False, infer_shape=True)

        def _pretrain_forward(i, h_tm0, h_tm1, g_predictions, s_predictions):
            x_t = model_input[:,i,:]
            gate, h_t0, h_t1 = g_recurrent_unit_forward(x_t, h_tm0, h_tm1)
            hidden_state, c_prev = tf.unstack(h_t1)
            g_predictions = g_predictions.write(i,hidden_state)
            s_predictions = s_predictions.write(i,c_prev)
            return i + 1, h_t0, h_t1, g_predictions, s_predictions

        _, _, _, hidden_outputs, state_outputs = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < max_frames,
            body=_pretrain_forward,
            loop_vars=(tf.constant(0, dtype=tf.int32), h1, h0,outputs_hidden,outputs_state))


        state_outputs = state_outputs.stack()
        state_outputs = tf.transpose(state_outputs, [1, 0, 2])
        hidden_outputs = hidden_outputs.stack()
        hidden_outputs = tf.transpose(hidden_outputs, [1, 0, 2])
        hidden_outputs = tf.reshape(hidden_outputs, [-1, max_frames, lstm_size])
        batch_size = tf.shape(model_input)[0]
        index_1 = tf.range(0, batch_size) * max_frames + (num_frames - 1)
        state_outputs = tf.gather(tf.reshape(state_outputs, [-1, hidden_dim]), index_1)

        return state_outputs, hidden_outputs

class LstmBigluModel(models.BaseModel):
    """Logistic model with L2 regularization."""

    def create_model(self, model_input, vocab_size, num_frames, l2_penalty=1e-8, **unused_params):
        """Creates a matrix regression model.

        Args:
          model_input: 'batch' x 'num_features' x 'num_methods' matrix of input features.
          vocab_size: The number of classes in the dataset.

        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes."""

        lstm_size = FLAGS.lstm_cells
        max_frames = model_input.get_shape().as_list()[1]
        emb_dim = model_input.get_shape().as_list()[2]
        hidden_dim = lstm_size
        model_input_reverse = tf.reverse_sequence(model_input,num_frames,seq_axis=1)

        with tf.variable_scope('lstm_forward'):
            g_recurrent_unit_forward = self.create_forward_recurrent_unit(emb_dim,hidden_dim,l2_penalty)
        with tf.variable_scope('lstm_backward'):
            g_recurrent_unit_backward = self.create_backward_recurrent_unit(emb_dim,hidden_dim,l2_penalty)

        h0 = tf.zeros([tf.shape(model_input)[0], hidden_dim])
        h0 = tf.stack([h0, h0])

        forward_outputs_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=max_frames,
            dynamic_size=False, infer_shape=True)
        forward_outputs_state = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=max_frames,
            dynamic_size=False, infer_shape=True)

        def _pretrain_forward(i, h_tm1, g_predictions, s_predictions):
            x_t = model_input_reverse[:,i,:]
            _, h_t1 = g_recurrent_unit_forward(x_t, h_tm1)
            hidden_state, c_prev = tf.unstack(h_t1)
            g_predictions = g_predictions.write(i, hidden_state)
            s_predictions = s_predictions.write(i,c_prev)
            return i + 1, h_t1, g_predictions, s_predictions

        _, _, forward_x_outputs, forward_state_outputs = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3: i < max_frames,
            body=_pretrain_forward,
            loop_vars=(tf.constant(0, dtype=tf.int32),  h0, forward_outputs_x, forward_outputs_state))

        forward_x_outputs = forward_x_outputs.stack()
        forward_state_outputs = forward_state_outputs.stack()
        batch_size = tf.shape(model_input)[0]
        index_1 = tf.range(0, batch_size) * max_frames + (num_frames - 1)
        forward_x_outputs = tf.transpose(forward_x_outputs, [1, 0, 2])
        x_outputs_reverse = tf.reverse_sequence(forward_x_outputs,num_frames,seq_axis=1)
        forward_state_outputs = tf.transpose(forward_state_outputs, [1, 0, 2])
        forward_state_outputs = tf.gather(tf.reshape(forward_state_outputs, [-1, hidden_dim]), index_1)

        h0 = tf.zeros([tf.shape(model_input)[0], hidden_dim])
        h0 = tf.stack([h0, h0])

        backward_outputs_state = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=max_frames,
            dynamic_size=False, infer_shape=True)

        def _pretrain_backward(i, h_tm1, s_predictions):
            x_t = model_input[:,i,:]
            y_t = x_outputs_reverse[:,i,:]
            _, h_t1 = g_recurrent_unit_backward(x_t, y_t, h_tm1)
            hidden_state, c_prev = tf.unstack(h_t1)
            s_predictions = s_predictions.write(i,c_prev)
            return i + 1, h_t1, s_predictions

        _, _, backward_state_outputs = control_flow_ops.while_loop(
            cond=lambda i, _1, _2: i < max_frames,
            body=_pretrain_backward,
            loop_vars=(tf.constant(0, dtype=tf.int32), h0, backward_outputs_state))

        backward_state_outputs = backward_state_outputs.stack()
        backward_state_outputs = tf.transpose(backward_state_outputs, [1, 0, 2])
        backward_state_outputs = tf.gather(tf.reshape(backward_state_outputs, [-1, hidden_dim]), index_1)
        moe_input = tf.concat((forward_state_outputs, backward_state_outputs), axis=1)

        final_probabilities = self.sub_moe(moe_input, vocab_size, scopename="backward")
        #probabilities_by_class = self.sub_moe(forward_state_outputs, vocab_size, scopename="forward")

        return {"predictions": final_probabilities}

    def create_forward_recurrent_unit(self,emb_dim,hidden_dim,l2_penalty):
        # Weights and Bias for input and hidden tensor
        Wi = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Wi")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wi))
        Ui = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=0.1), name="Ui")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Ui))
        bi = tf.Variable(tf.constant(0.1, shape=[1]), name="bi")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bi))

        Wf = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Wf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wf))
        Uf = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=0.1), name="Uf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uf))
        bf = tf.Variable(tf.constant(0.1, shape=[1]), name="bf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bf))

        Wog = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wog))
        Uog = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Uog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uog))
        bog = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bog))

        Wc = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wc))
        Uc = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Uc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uc))
        bc = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bc))

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)
            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, Wi) +
                tf.matmul(previous_hidden_state, Ui) + bi
            )
            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, Wf) +
                tf.matmul(previous_hidden_state, Uf) + bf
            )
            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, Wog) +
                tf.matmul(previous_hidden_state, Uog) + bog
            )
            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, Wc)+
                tf.matmul(previous_hidden_state, Uc) + bc
            )
            # Final Memory cell
            c = f * c_prev + i * c_
            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)
            return f, tf.stack([current_hidden_state, c])
        return unit

    def create_backward_recurrent_unit(self,emb_dim,hidden_dim,l2_penalty):
        # Weights and Bias for input and hidden tensor
        Wi = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Wi")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wi))
        Ui = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=0.1), name="Ui")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Ui))
        Vi = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=0.1), name="Vi")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Vi))
        bi = tf.Variable(tf.constant(0.1, shape=[1]), name="bi")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bi))

        Wf = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Wf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wf))
        Vf = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=0.1), name="Vf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Vf))
        Uf = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=0.1), name="Uf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uf))
        bf = tf.Variable(tf.constant(0.1, shape=[1]), name="bf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bf))

        Wog = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wog))
        Vog = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Vog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Vog))
        Uog = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Uog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uog))
        bog = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bog))

        Wc = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wc))
        Vc = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Vc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Vc))
        Uc = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Uc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uc))
        bc = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bc))


        def unit(x, y, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)
            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, Wi) + tf.matmul(y, Vi)+
                tf.matmul(previous_hidden_state, Ui) + bi
            )
            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, Wf) + tf.matmul(y, Vf) +
                tf.matmul(previous_hidden_state, Uf) + bf
            )
            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, Wog) + tf.matmul(y, Vog) +
                tf.matmul(previous_hidden_state, Uog) + bog
            )
            # New Memory Cell
            c_ = tf.tanh(
                tf.matmul(x, Wc) + tf.matmul(y, Vc) +
                tf.matmul(previous_hidden_state, Uc) + bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_
            # Current Hidden state
            current_hidden_state = o * tf.tanh(c)
            return f, tf.stack([current_hidden_state, c])
        return unit


    def sub_moe(self,model_input,
                vocab_size,
                num_mixtures=None,
                l2_penalty=1e-8,
                scopename="",
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

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates"+scopename)
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts"+scopename)

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures


        final_probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)

        final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                         [-1, vocab_size])

        return final_probabilities

class LstmBiglu2Model(models.BaseModel):
    """Logistic model with L2 regularization."""

    def create_model(self, model_input, vocab_size, num_frames, l2_penalty=1e-8, **unused_params):
        """Creates a matrix regression model.

        Args:
          model_input: 'batch' x 'num_features' x 'num_methods' matrix of input features.
          vocab_size: The number of classes in the dataset.

        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes."""

        lstm_size = FLAGS.lstm_cells
        max_frames = model_input.get_shape().as_list()[1]
        emb_dim = model_input.get_shape().as_list()[2]
        hidden_dim = lstm_size
        model_input_reverse = tf.reverse_sequence(model_input,num_frames,seq_axis=1)

        with tf.variable_scope('lstm_forward'):
            g_recurrent_unit_forward = self.create_forward_recurrent_unit(emb_dim,hidden_dim,l2_penalty)
        with tf.variable_scope('lstm_backward'):
            g_recurrent_unit_backward = self.create_backward_recurrent_unit(emb_dim,hidden_dim,l2_penalty)

        h0 = tf.zeros([tf.shape(model_input)[0], hidden_dim])
        h1 = tf.zeros([tf.shape(model_input)[0], emb_dim])
        h0 = tf.stack([h0, h0])
        h1 = tf.stack([h1, h1])

        forward_outputs_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=max_frames,
            dynamic_size=False, infer_shape=True)
        forward_outputs_state = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=max_frames,
            dynamic_size=False, infer_shape=True)

        def _pretrain_forward(i, h_tm0, h_tm1, g_predictions, s_predictions):
            x_t = model_input_reverse[:,i,:]
            gate, h_t0, h_t1 = g_recurrent_unit_forward(x_t, h_tm0, h_tm1)
            hidden_x, x_prev = tf.unstack(h_t0)
            hidden_state, c_prev = tf.unstack(h_t1)
            g_predictions = g_predictions.write(i,hidden_x)
            s_predictions = s_predictions.write(i,c_prev)
            return i + 1, h_t0, h_t1, g_predictions, s_predictions

        _, _, _, forward_x_outputs, forward_state_outputs = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < max_frames,
            body=_pretrain_forward,
            loop_vars=(tf.constant(0, dtype=tf.int32), h1, h0,forward_outputs_x,forward_outputs_state))

        forward_x_outputs = forward_x_outputs.stack()
        forward_state_outputs = forward_state_outputs.stack()
        batch_size = tf.shape(model_input)[0]
        index_1 = tf.range(0, batch_size) * max_frames + (num_frames - 1)
        forward_x_outputs = tf.transpose(forward_x_outputs, [1, 0, 2])
        x_outputs_reverse = tf.reverse_sequence(forward_x_outputs,num_frames,seq_axis=1)
        forward_state_outputs = tf.transpose(forward_state_outputs, [1, 0, 2])
        forward_state_outputs = tf.gather(tf.reshape(forward_state_outputs, [-1, hidden_dim]), index_1)

        h0 = tf.zeros([tf.shape(model_input)[0], hidden_dim])
        h1 = tf.zeros([tf.shape(model_input)[0], emb_dim])
        h0 = tf.stack([h0, h0])
        h1 = tf.stack([h1, h1])

        backward_outputs_state = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=max_frames,
            dynamic_size=False, infer_shape=True)

        def _pretrain_backward(i, h_tm0, h_tm1, s_predictions):
            x_t = model_input[:,i,:]
            y_t = x_outputs_reverse[:,i,:]
            gate, h_t0, h_t1 = g_recurrent_unit_backward(x_t, y_t, h_tm0, h_tm1)
            hidden_state, c_prev = tf.unstack(h_t1)
            s_predictions = s_predictions.write(i,c_prev)
            return i + 1, h_t0, h_t1, s_predictions

        _, _, _, backward_state_outputs = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3: i < max_frames,
            body=_pretrain_backward,
            loop_vars=(tf.constant(0, dtype=tf.int32), h1, h0,backward_outputs_state))

        backward_state_outputs = backward_state_outputs.stack()
        batch_size = tf.shape(model_input)[0]
        index_1 = tf.range(0, batch_size) * max_frames + (num_frames - 1)
        backward_state_outputs = tf.transpose(backward_state_outputs, [1, 0, 2])
        backward_state_outputs = tf.gather(tf.reshape(backward_state_outputs, [-1, hidden_dim]), index_1)
        moe_input = tf.concat((forward_state_outputs, backward_state_outputs), axis=1)

        final_probabilities = self.sub_moe(moe_input,vocab_size,scopename="backward")
        #probabilities_by_class = self.sub_moe(forward_state_outputs,vocab_size,scopename="forward")

        return {"predictions": final_probabilities}

    def create_forward_recurrent_unit(self,emb_dim,hidden_dim,l2_penalty):
        # Weights and Bias for input and hidden tensor
        Wi = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Wi")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wi))
        Ui = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=0.1), name="Ui")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Ui))
        Vi = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Vi")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Vi))
        bi = tf.Variable(tf.constant(0.1, shape=[1]), name="bi")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bi))

        Wf = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Wf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wf))
        Vf = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Vf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Vf))
        Uf = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=0.1), name="Uf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uf))
        bf = tf.Variable(tf.constant(0.1, shape=[1]), name="bf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bf))

        Wog = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wog))
        Vog = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Vog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Vog))
        Uog = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Uog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uog))
        bog = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bog))

        Wc = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wc))
        Vc = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Vc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Vc))
        Uc = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Uc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uc))
        bc = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bc))

        Wix = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Wix")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wix))
        Vix = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Vix")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Vix))
        Uix = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=0.1), name="Uix")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uix))
        bix = tf.Variable(tf.constant(0.1, shape=[1]), name="bix")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bix))

        Wfx = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Wfx")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wfx))
        Vfx = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Vfx")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Vfx))
        Ufx = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=0.1), name="Ufx")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Ufx))
        bfx = tf.Variable(tf.constant(0.1, shape=[1]), name="bfx")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bfx))

        def unit(x, hidden_memory_tm0, hidden_memory_tm1):
            previous_hidden_x, x_prev = tf.unstack(hidden_memory_tm0)
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)
            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, Wi) + tf.matmul(previous_hidden_x, Vi) +
                tf.matmul(previous_hidden_state, Ui) + bi
            )
            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, Wf) + tf.matmul(previous_hidden_x, Vf) +
                tf.matmul(previous_hidden_state, Uf) + bf
            )
            # Input Gate
            ix = tf.sigmoid(
                tf.matmul(x, Wix) + tf.matmul(previous_hidden_x, Vix) +
                tf.matmul(previous_hidden_state, Uix) + bix
            )
            # Forget Gate
            fx = tf.sigmoid(
                tf.matmul(x, Wfx) + tf.matmul(previous_hidden_x, Vfx) +
                tf.matmul(previous_hidden_state, Ufx) + bfx
            )
            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, Wog) + tf.matmul(previous_hidden_x, Vog) +
                tf.matmul(previous_hidden_state, Uog) + bog
            )
            # New Memory Cell
            c_ = tf.tanh(
                tf.matmul(x, Wc) + tf.matmul(previous_hidden_x, Vc) +
                tf.matmul(previous_hidden_state, Uc) + bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_
            current_x = fx * x_prev + ix * x
            current_hidden_x = tf.nn.l2_normalize(current_x, dim=1)
            # Current Hidden state
            current_hidden_state = o * tf.tanh(c)
            return f, tf.stack([current_hidden_x, current_x]), tf.stack([current_hidden_state, c])
        return unit

    def create_backward_recurrent_unit(self,emb_dim,hidden_dim,l2_penalty):
        # Weights and Bias for input and hidden tensor
        Wi = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Wi")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wi))
        Ui = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=0.1), name="Ui")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Ui))
        Vi = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Vi")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Vi))
        Xi = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Xi")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Xi))
        bi = tf.Variable(tf.constant(0.1, shape=[1]), name="bi")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bi))

        Wf = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Wf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wf))
        Vf = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Vf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Vf))
        Uf = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=0.1), name="Uf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uf))
        Xf = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Xf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Xf))
        bf = tf.Variable(tf.constant(0.1, shape=[1]), name="bf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bf))

        Wog = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wog))
        Vog = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Vog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Vog))
        Uog = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Uog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uog))
        Xog = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Xog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Xog))
        bog = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bog))

        Wc = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wc))
        Vc = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Vc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Vc))
        Uc = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Uc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uc))
        Xc = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Xc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Xc))
        bc = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bc))

        Wix = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Wix")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wix))
        Vix = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Vix")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Vix))
        Uix = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=0.1), name="Uix")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uix))
        Xix = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Xix")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Xix))
        bix = tf.Variable(tf.constant(0.1, shape=[1]), name="bix")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bix))

        Wfx = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Wfx")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wfx))
        Vfx = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Vfx")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Vfx))
        Ufx = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=0.1), name="Ufx")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Ufx))
        Xfx = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Xfx")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Xfx))
        bfx = tf.Variable(tf.constant(0.1, shape=[1]), name="bfx")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bfx))

        def unit(x, y, hidden_memory_tm0, hidden_memory_tm1):
            previous_hidden_x, x_prev = tf.unstack(hidden_memory_tm0)
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)
            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, Wi) + tf.matmul(previous_hidden_x, Vi) + tf.matmul(y, Xi) +
                tf.matmul(previous_hidden_state, Ui) + bi
            )
            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, Wf) + tf.matmul(previous_hidden_x, Vf) + tf.matmul(y, Xf) +
                tf.matmul(previous_hidden_state, Uf) + bf
            )
            # Input Gate
            ix = tf.sigmoid(
                tf.matmul(x, Wix) + tf.matmul(previous_hidden_x, Vix) + tf.matmul(y, Xix) +
                tf.matmul(previous_hidden_state, Uix) + bix
            )
            # Forget Gate
            fx = tf.sigmoid(
                tf.matmul(x, Wfx) + tf.matmul(previous_hidden_x, Vfx) + tf.matmul(y, Xfx) +
                tf.matmul(previous_hidden_state, Ufx) + bfx
            )
            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, Wog) + tf.matmul(previous_hidden_x, Vog) + tf.matmul(y, Xog) +
                tf.matmul(previous_hidden_state, Uog) + bog
            )
            # New Memory Cell
            c_ = tf.tanh(
                tf.matmul(x, Wc) + tf.matmul(previous_hidden_x, Vc) + tf.matmul(y, Xc) +
                tf.matmul(previous_hidden_state, Uc) + bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_
            current_x = fx * x_prev + ix * x
            current_hidden_x = tf.nn.l2_normalize(current_x, dim=1)
            # Current Hidden state
            current_hidden_state = o * tf.tanh(c)
            return f, tf.stack([current_hidden_x, current_x]), tf.stack([current_hidden_state, c])
        return unit

    def sub_moe(self,model_input,
                vocab_size,
                num_mixtures=None,
                l2_penalty=1e-8,
                scopename="",
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

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates"+scopename)
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts"+scopename)

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures


        final_probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)

        final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                         [-1, vocab_size])

        return final_probabilities

class LstmGateModel(models.BaseModel):
            """Logistic model with L2 regularization."""

            def create_model(self, model_input, vocab_size, num_frames, l2_penalty=1e-8, **unused_params):
                """Creates a matrix regression model.

                Args:
                  model_input: 'batch' x 'num_features' x 'num_methods' matrix of input features.
                  vocab_size: The number of classes in the dataset.

                Returns:
                  A dictionary with a tensor containing the probability predictions of the
                  model in the 'predictions' key. The dimensions of the tensor are
                  batch_size x num_classes."""

                lstm_size = FLAGS.lstm_cells
                max_frames = model_input.get_shape().as_list()[1]
                emb_dim = model_input.get_shape().as_list()[2]
                hidden_dim = lstm_size

                with tf.variable_scope('lstm_forward'):
                    g_recurrent_unit_forward = self.create_recurrent_unit(emb_dim,hidden_dim,l2_penalty)

                h0 = tf.zeros([tf.shape(model_input)[0], hidden_dim])
                h0 = tf.stack([h0, h0])

                outputs_gate = tensor_array_ops.TensorArray(
                    dtype=tf.float32, size=max_frames,
                    dynamic_size=False, infer_shape=True)
                outputs_state = tensor_array_ops.TensorArray(
                    dtype=tf.float32, size=max_frames,
                    dynamic_size=False, infer_shape=True)

                def _pretrain_forward(i,h_tm1,g_predictions,s_predictions):
                    x_t = model_input[:,i,:]
                    gate, h_t = g_recurrent_unit_forward(x_t, h_tm1)
                    hidden_state, c_prev = tf.unstack(h_t)
                    g_predictions = g_predictions.write(i,gate)
                    s_predictions = s_predictions.write(i,c_prev)
                    return i + 1, h_t, g_predictions, s_predictions

                _, _, gate_outputs, state_outputs = control_flow_ops.while_loop(
                    cond=lambda i, _1, _2, _3: i < max_frames,
                    body=_pretrain_forward,
                    loop_vars=(tf.constant(0, dtype=tf.int32),h0,outputs_gate,outputs_state))

                gate_outputs = gate_outputs.stack()
                state_outputs = state_outputs.stack()
                batch_size = tf.shape(model_input)[0]
                index_1 = tf.range(0, batch_size) * max_frames + (num_frames - 1)
                gate_outputs = tf.transpose(gate_outputs, [1, 0, 2])
                gate_outputs = tf.gather(tf.reshape(gate_outputs, [-1, hidden_dim]), index_1)
                state_outputs = tf.transpose(state_outputs, [1, 0, 2])
                state_outputs = tf.gather(tf.reshape(state_outputs, [-1, hidden_dim]), index_1)

                aggregated_model = getattr(video_level_models,
                                           FLAGS.video_level_classifier_model)
                return aggregated_model().create_model(
                    model_input=state_outputs,
                    vocab_size=vocab_size,
                    **unused_params)

            def create_recurrent_unit(self,emb_dim,hidden_dim,l2_penalty):
                # Weights and Bias for input and hidden tensor
                Wi = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Wi")
                tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wi))
                Ui = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=0.1), name="Ui")
                tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Ui))
                bi = tf.Variable(tf.constant(0.1, shape=[1]), name="bi")
                tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bi))

                Wf = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Wf")
                tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wf))
                Uf = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=0.1), name="Uf")
                tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uf))
                bf = tf.Variable(tf.constant(1.0, shape=[1]), name="bf")
                tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bf))

                Wog = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wog")
                tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wog))
                Uog = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Uog")
                tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uog))
                bog = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bog")
                tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bog))

                Wc = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wc")
                tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wc))
                Uc = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Uc")
                tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uc))
                bc = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bc")
                tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bc))

                def unit(x, hidden_memory_tm1):
                    previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)
                    # Input Gate
                    i = tf.sigmoid(
                        tf.matmul(x, Wi) +
                        tf.matmul(previous_hidden_state, Ui) + bi
                    )
                    # Forget Gate
                    f = tf.sigmoid(
                        tf.matmul(x, Wf) +
                        tf.matmul(previous_hidden_state, Uf) + bf
                    )
                    # Output Gate
                    o = tf.sigmoid(
                        tf.matmul(x, Wog) +
                        tf.matmul(previous_hidden_state, Uog) + bog
                    )
                    # New Memory Cell
                    c_ = tf.nn.tanh(
                        tf.matmul(x, Wc)+
                        tf.matmul(previous_hidden_state, Uc) + bc
                    )
                    # Final Memory cell
                    c = f * c_prev + i * c_
                    # Current Hidden state
                    current_hidden_state = o * tf.nn.tanh(c)
                    return f, tf.stack([current_hidden_state, c])
                return unit

class LstmGateMultilayerModel(models.BaseModel):
    """Logistic model with L2 regularization."""

    def create_model(self, model_input, vocab_size, num_frames, l2_penalty=1e-8, **unused_params):
        """Creates a matrix regression model.

        Args:
          model_input: 'batch' x 'num_features' x 'num_methods' matrix of input features.
          vocab_size: The number of classes in the dataset.

        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes."""

        lstm_size = FLAGS.lstm_cells
        number_of_layers = FLAGS.lstm_layers

        hidden_outputs = model_input
        state_outputs = []
        for i in range(number_of_layers):
            state_output, hidden_outputs = self.rnn_gate(hidden_outputs, lstm_size, num_frames, sub_scope="lstm_lsyer%d" % i)
            state_outputs.append(state_output)

        state_outputs = tf.concat(state_outputs,axis=1)
        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=state_outputs,
            vocab_size=vocab_size,
            **unused_params)

    def rnn_gate(self, model_input, lstm_size, num_frames, l2_penalty=1e-8, sub_scope="", **unused_params):
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
        ## Batch normalize the input

        max_frames = model_input.get_shape().as_list()[1]
        emb_dim = model_input.get_shape().as_list()[2]
        hidden_dim = lstm_size

        with tf.variable_scope(sub_scope+'lstm_forward'):
            g_recurrent_unit_forward = self.create_recurrent_unit(emb_dim,hidden_dim,l2_penalty)

        h0 = tf.zeros([tf.shape(model_input)[0], hidden_dim])
        h0 = tf.stack([h0, h0])

        outputs_hidden = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=max_frames,
            dynamic_size=False, infer_shape=True)
        outputs_state = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=max_frames,
            dynamic_size=False, infer_shape=True)

        def _pretrain_forward(i,h_tm1,s_predictions,h_predictions):
            x_t = model_input[:,i,:]
            gate, h_t = g_recurrent_unit_forward(x_t, h_tm1)
            hidden_state, c_prev = tf.unstack(h_t)
            h_predictions = h_predictions.write(i,hidden_state)
            s_predictions = s_predictions.write(i,c_prev)
            return i + 1, h_t, s_predictions, h_predictions

        _, _, state_outputs, hidden_outputs = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3: i < max_frames,
            body=_pretrain_forward,
            loop_vars=(tf.constant(0, dtype=tf.int32), h0, outputs_state, outputs_hidden),
            swap_memory=True)

        state_outputs = state_outputs.stack()
        state_outputs = tf.transpose(state_outputs, [1, 0, 2])
        hidden_outputs = hidden_outputs.stack()
        hidden_outputs = tf.transpose(hidden_outputs, [1, 0, 2])
        hidden_outputs = tf.reshape(hidden_outputs, [-1, max_frames, lstm_size])
        batch_size = tf.shape(model_input)[0]
        index_1 = tf.range(0, batch_size) * max_frames + (num_frames - 1)
        state_outputs = tf.gather(tf.reshape(state_outputs, [-1, hidden_dim]), index_1)

        return state_outputs, hidden_outputs

    def create_recurrent_unit(self,emb_dim,hidden_dim,l2_penalty):
        # Weights and Bias for input and hidden tensor
        Wi = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Wi")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wi))
        Ui = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=0.1), name="Ui")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Ui))
        bi = tf.Variable(tf.constant(0.1, shape=[1]), name="bi")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bi))

        Wf = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Wf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wf))
        Uf = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=0.1), name="Uf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uf))
        bf = tf.Variable(tf.constant(1.0, shape=[1]), name="bf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bf))

        Wog = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wog))
        Uog = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Uog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uog))
        bog = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bog))

        Wc = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wc))
        Uc = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Uc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uc))
        bc = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bc))

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)
            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, Wi) +
                tf.matmul(previous_hidden_state, Ui) + bi
            )
            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, Wf) +
                tf.matmul(previous_hidden_state, Uf) + bf
            )
            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, Wog) +
                tf.matmul(previous_hidden_state, Uog) + bog
            )
            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, Wc)+
                tf.matmul(previous_hidden_state, Uc) + bc
            )
            # Final Memory cell
            c = f * c_prev + i * c_
            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)
            return f, tf.stack([current_hidden_state, c])
        return unit

class LstmQuickMemoryModel(models.BaseModel):
    """Logistic model with L2 regularization."""

    def create_model(self, model_input, vocab_size, num_frames, l2_penalty=1e-8, **unused_params):
        """Creates a matrix regression model.

        Args:
          model_input: 'batch' x 'num_features' x 'num_methods' matrix of input features.
          vocab_size: The number of classes in the dataset.

        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes."""

        lstm_size = FLAGS.lstm_cells
        max_frames = model_input.get_shape().as_list()[1]
        emb_dim = model_input.get_shape().as_list()[2]
        hidden_dim = lstm_size

        with tf.variable_scope('lstm_forward'):
            g_recurrent_unit_forward = self.create_recurrent_unit(emb_dim,hidden_dim,l2_penalty)

        h0 = tf.zeros([tf.shape(model_input)[0], hidden_dim])
        h0 = tf.stack([h0, h0, h0])

        outputs_state = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=max_frames,
            dynamic_size=False, infer_shape=True)

        def _pretrain_forward(i, h_tm1, s_predictions):
            x_t = model_input[:,i,:]
            gate, h_t1 = g_recurrent_unit_forward(x_t, h_tm1)
            hidden_state, c_prev, m_prev = tf.unstack(h_t1)
            s_predictions = s_predictions.write(i, c_prev + m_prev)
            return i + 1, h_t1, s_predictions

        _, _, state_outputs = control_flow_ops.while_loop(
            cond=lambda i, _1, _2: i < max_frames,
            body=_pretrain_forward,
            loop_vars=(tf.constant(0, dtype=tf.int32), h0,outputs_state))

        state_outputs = state_outputs.stack()
        batch_size = tf.shape(model_input)[0]
        index_1 = tf.range(0, batch_size) * max_frames + (num_frames - 1)
        state_outputs = tf.transpose(state_outputs, [1, 0, 2])
        state_outputs = tf.gather(tf.reshape(state_outputs, [-1, hidden_dim]), index_1)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=state_outputs,
            vocab_size=vocab_size,
            **unused_params)

    def create_recurrent_unit(self,emb_dim,hidden_dim,l2_penalty):
        # Weights and Bias for input and hidden tensor
        Wi = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Wi")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wi))
        Ui = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=0.1), name="Ui")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Ui))
        bi = tf.Variable(tf.constant(0.1, shape=[1]), name="bi")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bi))

        Wf = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Wf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wf))
        Uf = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=0.1), name="Uf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uf))
        bf = tf.Variable(tf.constant(0.1, shape=[1]), name="bf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bf))


        Wog = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wog))
        Uog = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Uog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uog))
        bog = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bog))

        Wc = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wc))
        Uc = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Uc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uc))
        bc = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bc))


        Wom = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wom")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wom))
        Uom = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Uom")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uom))
        bom = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bom")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bom))

        Wfm = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wfm")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wfm))
        Ufm = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Ufm")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Ufm))
        bfm = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bfm")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bfm))

        Wm = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wm")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wm))
        Um = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Um")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Um))
        bm = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bm")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bm))


        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev, m_prev = tf.unstack(hidden_memory_tm1)
            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, Wi) +
                tf.matmul(previous_hidden_state, Ui) + bi
            )
            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, Wf) +
                tf.matmul(previous_hidden_state, Uf) + bf
            )
            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, Wog) +
                tf.matmul(previous_hidden_state, Uog) + bog
            )
            # long Gate
            im = tf.sigmoid(
                tf.matmul(x, Wom) +
                tf.matmul(previous_hidden_state, Uom) + bom
            )
            # long Gate
            fm = tf.sigmoid(
                tf.matmul(x, Wfm) +
                tf.matmul(previous_hidden_state, Ufm) + bfm
            )
            # New Memory Cell
            c_ = tf.tanh(
                tf.matmul(x, Wc) +
                tf.matmul(previous_hidden_state, Uc) + bc
            )

            # New Memory Cell
            m_ = tf.tanh(
                tf.matmul(x, Wm) +
                tf.matmul(previous_hidden_state, Um) + bm
            )

            # Final Memory cell
            m = im * m_ + fm * m_prev
            c = f * c_prev + i * c_
            # Current Hidden state
            current_hidden_state = o * tf.tanh(c + m)
            return f, tf.stack([current_hidden_state, c, m])
        return unit

class LstmLinearOutputModel(models.BaseModel):
    """Logistic model with L2 regularization."""

    def create_model(self, model_input, vocab_size, num_frames, l2_penalty=1e-8, **unused_params):
        """Creates a matrix regression model.

        Args:
          model_input: 'batch' x 'num_features' x 'num_methods' matrix of input features.
          vocab_size: The number of classes in the dataset.

        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes."""

        lstm_size = FLAGS.lstm_cells
        max_frames = model_input.get_shape().as_list()[1]
        emb_dim = model_input.get_shape().as_list()[2]
        hidden_dim = lstm_size

        with tf.variable_scope('lstm_forward'):
            g_recurrent_unit_forward = self.create_recurrent_unit(emb_dim,hidden_dim,l2_penalty)

        h0 = tf.zeros([tf.shape(model_input)[0], hidden_dim])
        h0 = tf.stack([h0, h0])

        outputs_gate = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=max_frames,
            dynamic_size=False, infer_shape=True)
        outputs_state = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=max_frames,
            dynamic_size=False, infer_shape=True)

        def _pretrain_forward(i,h_tm1,g_predictions,s_predictions):
            x_t = model_input[:,i,:]
            gate, h_t = g_recurrent_unit_forward(x_t, h_tm1)
            hidden_state, c_prev = tf.unstack(h_t)
            g_predictions = g_predictions.write(i,gate)
            s_predictions = s_predictions.write(i,c_prev)
            return i + 1, h_t, g_predictions, s_predictions

        _, _, gate_outputs, state_outputs = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3: i < max_frames,
            body=_pretrain_forward,
            loop_vars=(tf.constant(0, dtype=tf.int32),h0,outputs_gate,outputs_state))

        gate_outputs = gate_outputs.stack()
        state_outputs = state_outputs.stack()
        batch_size = tf.shape(model_input)[0]
        index_1 = tf.range(0, batch_size) * max_frames + (num_frames - 1)
        gate_outputs = tf.transpose(gate_outputs, [1, 0, 2])
        gate_outputs = tf.gather(tf.reshape(gate_outputs, [-1, hidden_dim]), index_1)
        state_outputs = tf.transpose(state_outputs, [1, 0, 2])
        state_outputs = tf.gather(tf.reshape(state_outputs, [-1, hidden_dim]), index_1)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=state_outputs,
            vocab_size=vocab_size,
            **unused_params)

    def create_recurrent_unit(self,emb_dim,hidden_dim,l2_penalty):
        # Weights and Bias for input and hidden tensor
        Wi = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Wi")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wi))
        Ui = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=0.1), name="Ui")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Ui))
        bi = tf.Variable(tf.constant(0.1, shape=[1]), name="bi")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bi))

        Wf = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Wf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wf))
        Uf = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=0.1), name="Uf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uf))
        bf = tf.Variable(tf.constant(0.1, shape=[1]), name="bf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bf))

        Wog = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wog))
        Uog = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Uog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uog))
        bog = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bog))

        Wc = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wc))
        Uc = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Uc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uc))
        bc = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bc))

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)
            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, Wi) +
                tf.matmul(previous_hidden_state, Ui) + bi
            )
            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, Wf) +
                tf.matmul(previous_hidden_state, Uf) + bf
            )
            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, Wog) +
                tf.matmul(previous_hidden_state, Uog) + bog
            )
            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, Wc)+
                tf.matmul(previous_hidden_state, Uc) + bc
            )
            # Final Memory cell
            c = f * c_prev + i * c_
            # Current Hidden state
            current_hidden_state = o * tf.nn.l2_normalize(c,dim=1)
            return f, tf.stack([current_hidden_state, c])
        return unit

class LstmLinearOutput2Model(models.BaseModel):
    """Logistic model with L2 regularization."""

    def create_model(self, model_input, vocab_size, num_frames, l2_penalty=1e-8, **unused_params):
        """Creates a matrix regression model.

        Args:
          model_input: 'batch' x 'num_features' x 'num_methods' matrix of input features.
          vocab_size: The number of classes in the dataset.

        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes."""

        lstm_size = FLAGS.lstm_cells
        max_frames = model_input.get_shape().as_list()[1]
        emb_dim = model_input.get_shape().as_list()[2]
        hidden_dim = lstm_size

        with tf.variable_scope('lstm_forward'):
            g_recurrent_unit_forward = self.create_recurrent_unit(emb_dim,hidden_dim,l2_penalty)

        h0 = tf.zeros([tf.shape(model_input)[0], hidden_dim])
        h0 = tf.stack([h0, h0])

        outputs_gate = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=max_frames,
            dynamic_size=False, infer_shape=True)
        outputs_state = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=max_frames,
            dynamic_size=False, infer_shape=True)

        def _pretrain_forward(i,h_tm1,g_predictions,s_predictions):
            x_t = model_input[:,i,:]
            gate, h_t = g_recurrent_unit_forward(x_t, h_tm1)
            hidden_state, c_prev = tf.unstack(h_t)
            g_predictions = g_predictions.write(i,gate)
            s_predictions = s_predictions.write(i,c_prev)
            return i + 1, h_t, g_predictions, s_predictions

        _, _, gate_outputs, state_outputs = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3: i < max_frames,
            body=_pretrain_forward,
            loop_vars=(tf.constant(0, dtype=tf.int32),h0,outputs_gate,outputs_state))

        gate_outputs = gate_outputs.stack()
        state_outputs = state_outputs.stack()
        batch_size = tf.shape(model_input)[0]
        index_1 = tf.range(0, batch_size) * max_frames + (num_frames - 1)
        gate_outputs = tf.transpose(gate_outputs, [1, 0, 2])
        gate_outputs = tf.gather(tf.reshape(gate_outputs, [-1, hidden_dim]), index_1)
        state_outputs = tf.transpose(state_outputs, [1, 0, 2])
        state_outputs = tf.gather(tf.reshape(state_outputs, [-1, hidden_dim]), index_1)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=state_outputs,
            vocab_size=vocab_size,
            **unused_params)

    def create_recurrent_unit(self,emb_dim,hidden_dim,l2_penalty):
        # Weights and Bias for input and hidden tensor
        Wi = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wi")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wi))
        Ui = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Ui")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Ui))
        bi = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bi")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bi))

        Wf = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wf))
        Uf = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Uf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uf))
        bf = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bf))

        Wog = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wog))
        Uog = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Uog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uog))
        bog = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bog")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bog))

        Wc = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wc))
        Uc = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Uc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uc))
        bc = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bc))

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)
            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, Wi) +
                tf.matmul(previous_hidden_state, Ui) + bi
            )
            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, Wf) +
                tf.matmul(previous_hidden_state, Uf) + bf + 1.0
            )
            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, Wog) +
                tf.matmul(previous_hidden_state, Uog) + bog
            )
            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, Wc) +
                tf.matmul(previous_hidden_state, Uc) + bc
            )
            # Final Memory cell
            c = f * c_prev + i * c_
            # Current Hidden state
            current_hidden_state = o * tf.nn.l2_normalize(c,dim=1)
            return f, tf.stack([current_hidden_state, c])
        return unit

class LstmNoOutputModel(models.BaseModel):
    """Logistic model with L2 regularization."""

    def create_model(self, model_input, vocab_size, num_frames, l2_penalty=1e-8, **unused_params):
        """Creates a matrix regression model.

        Args:
          model_input: 'batch' x 'num_features' x 'num_methods' matrix of input features.
          vocab_size: The number of classes in the dataset.

        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes."""

        lstm_size = FLAGS.lstm_cells
        max_frames = model_input.get_shape().as_list()[1]
        emb_dim = model_input.get_shape().as_list()[2]
        hidden_dim = lstm_size

        with tf.variable_scope('lstm_forward'):
            g_recurrent_unit_forward = self.create_recurrent_unit(emb_dim,hidden_dim,l2_penalty)

        h0 = tf.zeros([tf.shape(model_input)[0], hidden_dim])
        h0 = tf.stack([h0, h0])

        outputs_gate = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=max_frames,
            dynamic_size=False, infer_shape=True)
        outputs_state = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=max_frames,
            dynamic_size=False, infer_shape=True)

        def _pretrain_forward(i,h_tm1,g_predictions,s_predictions):
            x_t = model_input[:,i,:]
            gate, h_t = g_recurrent_unit_forward(x_t, h_tm1)
            hidden_state, c_prev = tf.unstack(h_t)
            g_predictions = g_predictions.write(i,gate)
            s_predictions = s_predictions.write(i,c_prev)
            return i + 1, h_t, g_predictions, s_predictions

        _, _, gate_outputs, state_outputs = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3: i < max_frames,
            body=_pretrain_forward,
            loop_vars=(tf.constant(0, dtype=tf.int32),h0,outputs_gate,outputs_state))

        gate_outputs = gate_outputs.stack()
        state_outputs = state_outputs.stack()
        batch_size = tf.shape(model_input)[0]
        index_1 = tf.range(0, batch_size) * max_frames + (num_frames - 1)
        gate_outputs = tf.transpose(gate_outputs, [1, 0, 2])
        gate_outputs = tf.gather(tf.reshape(gate_outputs, [-1, hidden_dim]), index_1)
        state_outputs = tf.transpose(state_outputs, [1, 0, 2])
        state_outputs = tf.gather(tf.reshape(state_outputs, [-1, hidden_dim]), index_1)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=state_outputs,
            vocab_size=vocab_size,
            **unused_params)

    def create_recurrent_unit(self,emb_dim,hidden_dim,l2_penalty):
        # Weights and Bias for input and hidden tensor
        Wi = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Wi")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wi))
        Ui = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=0.1), name="Ui")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Ui))
        bi = tf.Variable(tf.constant(0.1, shape=[1]), name="bi")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bi))

        Wf = tf.Variable(tf.truncated_normal([emb_dim, 1], stddev=0.1), name="Wf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wf))
        Uf = tf.Variable(tf.truncated_normal([hidden_dim, 1], stddev=0.1), name="Uf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uf))
        bf = tf.Variable(tf.constant(0.1, shape=[1]), name="bf")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bf))

        Wc = tf.Variable(tf.truncated_normal([emb_dim, hidden_dim], stddev=0.1), name="Wc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Wc))
        Uc = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.1), name="Uc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(Uc))
        bc = tf.Variable(tf.constant(0.1, shape=[hidden_dim]), name="bc")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(bc))

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)
            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, Wi) +
                tf.matmul(previous_hidden_state, Ui) + bi
            )
            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, Wf) +
                tf.matmul(previous_hidden_state, Uf) + bf
            )
            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, Wc)+
                tf.matmul(previous_hidden_state, Uc) + bc
            )
            # Final Memory cell
            c = f * c_prev + i * c_
            # Current Hidden state
            current_hidden_state = tf.nn.l2_normalize(c,dim=1)
            return f, tf.stack([current_hidden_state, c])
        return unit

class LstmModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
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
      aggregated_model = getattr(video_level_models,
                                 FLAGS.video_level_classifier_model)
      return aggregated_model().create_model(
          model_input=state_c,
          vocab_size=vocab_size,
          **unused_params)

class LstmDiffModel(models.BaseModel):

    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
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

        model_mean = tf.reduce_sum(model_input*frames_bool,axis=1,keep_dims=True)/tf.reduce_sum(frames_bool,axis=1,keep_dims=True)
        model_input = (model_input - model_mean)*frames_bool
        model_input = tf.nn.l2_normalize(model_input, 2)

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
        model_mean = tf.nn.l2_normalize(tf.reshape(model_mean,[-1,shape[2]]), dim=1)
        rnn_out = tf.nn.l2_normalize(state_c, dim=1)
        model_output = tf.concat((model_mean,rnn_out),axis=1)
        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=model_output,
            vocab_size=vocab_size,
            **unused_params)

class LstmFramesModel(models.BaseModel):

    def sub_moe(self,model_input,
                     vocab_size,
                     num_mixtures=None,
                     l2_penalty=1e-8,
                     scopename="",
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
            scope="gates"+scopename)
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts"+scopename)

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
        with tf.name_scope("loss_frames"):
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

class LstmFrames2Model(models.BaseModel):

    def sub_moe(self,model_input,
                vocab_size,
                num_mixtures=None,
                l2_penalty=1e-8,
                scopename="",
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

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates"+scopename)
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts"+scopename)

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures


        final_probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)

        final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                                        [-1, vocab_size])

        return final_probabilities

    def calculate_loss(self, predictions, labels, **unused_params):
        with tf.name_scope("loss_frames"):
            epsilon = 10e-6
            float_labels = tf.cast(labels, tf.float32)
            cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + (
                1 - float_labels) * tf.log(1 - predictions + epsilon)

        return tf.reduce_sum(cross_entropy_loss, axis=2)


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

        model_mean = tf.reduce_sum(model_input*frames_bool, axis=1)/tf.reduce_sum(frames_bool, axis=1)
        video_probabilities = self.sub_moe(model_mean, vocab_size, scopename="_video")

        unit_layer_1 = FLAGS.stride_size
        unit_layer_2 = shape[1]//FLAGS.stride_size
        model_input = model_input[:,:unit_layer_1*unit_layer_2,:]

        model_input_1 = tf.reshape(model_input,[-1,unit_layer_1,shape[2]])
        sigmoid_input = tf.reduce_mean(model_input_1,axis=1)
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
        state_out = tf.concat(map(lambda x: x.c, state), axis=1)
        probabilities_by_batch = slim.fully_connected(
            sigmoid_input,
            vocab_size,
            activation_fn=tf.nn.sigmoid,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="expert_activations")

        probabilities_by_frame = tf.reshape(probabilities_by_batch,[-1,unit_layer_2,vocab_size])

        importance_by_frame = self.calculate_loss(probabilities_by_frame, tf.reshape(video_probabilities,[-1,1,vocab_size]))
        _, index_topk = tf.nn.top_k(importance_by_frame, k=FLAGS.moe_num_extend)

        batch_size = tf.shape(model_input)[0]
        batch_index = tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, FLAGS.moe_num_extend])
        index = tf.reshape(batch_index*unit_layer_2 + index_topk,[-1])
        moe_input = tf.gather(state_out, index)
        frame_probabilities = self.sub_moe(moe_input,vocab_size,scopename="_frame")
        result = {}
        result["prediction_frames"] = frame_probabilities
        result["prediction_prepare_frames"] = tf.reshape(probabilities_by_frame,[-1,vocab_size])
        result["prediction_prepare_video"] = video_probabilities
        result["predictions"] = tf.reduce_mean(tf.reshape(frame_probabilities,[-1,FLAGS.moe_num_extend,vocab_size]),axis=1)
        return result

class LstmFrames3Model(models.BaseModel):

    def sub_moe(self,model_input,
                vocab_size,
                num_mixtures=None,
                l2_penalty=1e-8,
                scopename="",
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

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates"+scopename)
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts"+scopename)

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures


        final_probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)

        final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                         [-1, vocab_size])

        return final_probabilities


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
        num_extend = FLAGS.moe_num_extend - 1


        unit_layer_1 = FLAGS.stride_size
        unit_layer_2 = shape[1]//FLAGS.stride_size
        model_input = model_input[:,:unit_layer_1*unit_layer_2,:]
        num_frames = tf.maximum(num_frames//FLAGS.stride_size//num_extend, 1)

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
        state_out = tf.concat(map(lambda x: x.c, state), axis=1)
        random_vector = tf.random_uniform([lstm_size*number_of_layers,1],-1,1)
        random_out = tf.reshape(tf.matmul(state_out,random_vector),[-1,unit_layer_2])
        _, indexes = tf.nn.top_k(random_out, sorted=True, k=unit_layer_2)
        flat_state_out = tf.reshape(state_out,[-1,unit_layer_2,lstm_size*number_of_layers])
        divide_inputs = []
        length = unit_layer_2//num_extend
        frame_bool = tf.reshape(tf.sequence_mask(num_frames,maxlen=length,dtype=tf.float32),[-1,length,1])
        for i in range(num_extend):
            begin_frames = tf.reshape(num_frames*i,[-1,1])
            frames_index = tf.reshape(tf.range(length),[1,length])
            frames_index = begin_frames + frames_index
            batch_size = tf.shape(model_input)[0]
            batch_index = tf.tile(
                tf.expand_dims(tf.range(batch_size), 1), [1, length])
            index = tf.stack([batch_index, tf.cast(frames_index,dtype=tf.int32)], 2)
            divide_index = tf.gather_nd(indexes, index)
            batch_features_index = tf.stack([batch_index, tf.cast(divide_index,dtype=tf.int32)], 2)
            divide_feature = tf.gather_nd(flat_state_out, batch_features_index)
            divide_feature = tf.reduce_sum(divide_feature*frame_bool,axis=1)/(tf.reduce_sum(frame_bool,axis=1)+1e-6)
            divide_inputs.append(divide_feature)

        moe_input = tf.reshape(tf.stack(divide_inputs,axis=1),[-1,lstm_size*number_of_layers])
        frame_probabilities = self.sub_moe(moe_input,vocab_size,scopename="_frame")
        result = {}
        flat_frame_probabilities = tf.reshape(frame_probabilities,[-1,num_extend,vocab_size])

        softmax_activations = slim.fully_connected(
            moe_input,
            vocab_size,
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="softmax")
        softmax_distribution = tf.nn.softmax(tf.reshape(
            softmax_activations,
            [-1,num_extend,vocab_size]),dim=1)  # (Batch * #Labels) x (num_mixtures + 1)

        final_probabilities = tf.reduce_sum(
            softmax_distribution * flat_frame_probabilities, axis=1)

        result["predictions"] = final_probabilities

        result["prediction_frames"] = tf.reshape(tf.concat((flat_frame_probabilities,
                                                 tf.reshape(final_probabilities,[-1,1,vocab_size])),axis=1),[-1,vocab_size])
        return result

class LstmMultiscaleModel(models.BaseModel):

    def cnn(self,
            model_input,
            l2_penalty=1e-8,
            num_filters = [1024, 1024, 1024],
            filter_sizes = [1,2,3],
            sub_scope="",
            **unused_params):
        max_frames = model_input.get_shape().as_list()[1]
        num_features = model_input.get_shape().as_list()[2]

        shift_inputs = []
        for i in range(max(filter_sizes)):
            if i == 0:
                shift_inputs.append(model_input)
            else:
                shift_inputs.append(tf.pad(model_input, paddings=[[0,0],[i,0],[0,0]])[:,:max_frames,:])

        cnn_outputs = []
        for nf, fs in zip(num_filters, filter_sizes):
            sub_input = tf.concat(shift_inputs[:fs], axis=2)
            sub_filter = tf.get_variable(sub_scope+"cnn-filter-len%d"%fs,
                                         shape=[num_features*fs, nf], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                         regularizer=tf.contrib.layers.l2_regularizer(l2_penalty))
            cnn_outputs.append(tf.einsum("ijk,kl->ijl", sub_input, sub_filter))

        cnn_output = tf.concat(cnn_outputs, axis=2)
        cnn_output = slim.batch_norm(
            cnn_output,
            center=True,
            scale=True,
            is_training=FLAGS.train,
            scope=sub_scope+"cluster_bn")
        return cnn_output, max_frames

    def sub_moe(self,model_input,
                vocab_size,
                num_mixtures=None,
                l2_penalty=1e-8,
                scopename="",
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

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates"+scopename)
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts"+scopename)

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures


        final_probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)

        final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                         [-1, vocab_size])

        return final_probabilities

    def rnn(self, model_input, lstm_size, num_frames,sub_scope="", **unused_params):
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
        ## Batch normalize the input
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=True)
                for _ in range(1)
                ],
            state_is_tuple=True)
        with tf.variable_scope("RNN-"+sub_scope):
            outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                               sequence_length=num_frames,
                                               swap_memory=True,
                                               dtype=tf.float32)

        state_out = tf.concat(map(lambda x: x.c, state), axis=1)

        return state_out

    def create_model(self, model_input, vocab_size, num_frames, l2_penalty=1e-8, **unused_params):

        num_extend = FLAGS.moe_num_extend
        num_layers = num_extend
        lstm_size = FLAGS.lstm_cells
        pool_size=2
        cnn_input = model_input
        num_filters=[256,256,512]
        filter_sizes=[1,2,3]
        features_size = sum(num_filters)
        final_probilities = []
        moe_inputs = []
        for layer in range(num_layers):
            cnn_output, num_t = self.cnn(cnn_input, num_filters=num_filters, filter_sizes=filter_sizes, sub_scope="cnn%d"%(layer+1))
            cnn_output = tf.nn.relu(cnn_output)
            cnn_multiscale = self.rnn(cnn_output,lstm_size, num_frames,sub_scope="rnn%d"%(layer+1))
            moe_inputs.append(cnn_multiscale)
            final_probility = self.sub_moe(cnn_multiscale,vocab_size,scopename="moe%d"%(layer+1))
            final_probilities.append(final_probility)
            num_t = pool_size*(num_t//pool_size)
            cnn_output = tf.reshape(cnn_output[:,:num_t,:],[-1,num_t//pool_size,pool_size,features_size])
            cnn_input = tf.reduce_max(cnn_output, axis=2)
            num_frames = tf.maximum(num_frames//pool_size,1)

        final_probilities = tf.stack(final_probilities,axis=1)
        moe_inputs = tf.stack(moe_inputs,axis=1)
        weight2d = tf.get_variable("ensemble_weight2d",
                                   shape=[num_extend, features_size, vocab_size],
                                   regularizer=slim.l2_regularizer(1.0e-8))
        weight = tf.nn.softmax(tf.einsum("aij,ijk->aik", moe_inputs, weight2d), dim=1)
        result = {}
        result["prediction_frames"] = tf.reshape(final_probilities,[-1,vocab_size])
        result["predictions"] = tf.reduce_sum(final_probilities*weight,axis=1)
        return result

class LstmMultiscaleDitillChainModel(models.BaseModel):

    def cnn(self,
            model_input,
            l2_penalty=1e-8,
            num_filters = [1024, 1024, 1024],
            filter_sizes = [1,2,3],
            sub_scope="",
            **unused_params):
        max_frames = model_input.get_shape().as_list()[1]
        num_features = model_input.get_shape().as_list()[2]

        shift_inputs = []
        for i in range(max(filter_sizes)):
            if i == 0:
                shift_inputs.append(model_input)
            else:
                shift_inputs.append(tf.pad(model_input, paddings=[[0,0],[i,0],[0,0]])[:,:max_frames,:])

        cnn_outputs = []
        for nf, fs in zip(num_filters, filter_sizes):
            sub_input = tf.concat(shift_inputs[:fs], axis=2)
            sub_filter = tf.get_variable(sub_scope+"cnn-filter-len%d"%fs,
                                         shape=[num_features*fs, nf], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                         regularizer=tf.contrib.layers.l2_regularizer(l2_penalty))
            cnn_outputs.append(tf.einsum("ijk,kl->ijl", sub_input, sub_filter))

        cnn_output = tf.concat(cnn_outputs, axis=2)
        cnn_output = slim.batch_norm(
            cnn_output,
            center=True,
            scale=True,
            is_training=FLAGS.train,
            scope=sub_scope+"cluster_bn")
        return cnn_output, max_frames

    def sub_moe(self,model_input,
                vocab_size,
                distill_labels=None,
                num_mixtures=None,
                l2_penalty=1e-8,
                scopename="",
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
        class_size = 256
        if distill_labels is not None:
            class_input = slim.fully_connected(
                distill_labels,
                class_size,
                activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="class_inputs"+scopename)
            model_input = tf.concat((model_input,class_input),axis=1)
        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates"+scopename)
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts"+scopename)

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures


        final_probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)

        final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                         [-1, vocab_size])

        return final_probabilities

    def rnn(self, model_input, lstm_size, num_frames,sub_scope="", **unused_params):
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
        ## Batch normalize the input
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=True)
                for _ in range(1)
                ],
            state_is_tuple=True)
        with tf.variable_scope("RNN-"+sub_scope):
            outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                               sequence_length=num_frames,
                                               swap_memory=True,
                                               dtype=tf.float32)

        state_out = tf.concat(map(lambda x: x.c, state), axis=1)

        return state_out

    def create_model(self, model_input, vocab_size, num_frames, distill_labels=None, l2_penalty=1e-8, **unused_params):

        num_extend = FLAGS.moe_num_extend
        num_layers = num_extend
        lstm_size = FLAGS.lstm_cells
        pool_size = 2
        cnn_input = model_input
        cnn_size = FLAGS.cnn_cells
        num_filters = [cnn_size, cnn_size, cnn_size*2]
        filter_sizes = [1, 2, 3]
        features_size = sum(num_filters)
        final_probilities = []
        moe_inputs = []

        for layer in range(num_layers):
            cnn_output, num_t = self.cnn(cnn_input, num_filters=num_filters, filter_sizes=filter_sizes, sub_scope="cnn%d"%(layer+1))
            cnn_output = tf.nn.relu(cnn_output)
            cnn_multiscale = self.rnn(cnn_output,lstm_size, num_frames,sub_scope="rnn%d"%(layer+1))
            moe_inputs.append(cnn_multiscale)
            final_probility = self.sub_moe(cnn_multiscale,vocab_size,distill_labels=distill_labels, scopename="moe%d"%(layer+1))
            final_probilities.append(final_probility)
            num_t = pool_size*(num_t//pool_size)
            cnn_output = tf.reshape(cnn_output[:,:num_t,:],[-1,num_t//pool_size,pool_size,features_size])
            cnn_input = tf.reduce_max(cnn_output, axis=2)
            num_frames = tf.maximum(num_frames//pool_size,1)

        final_probilities = tf.stack(final_probilities,axis=1)
        moe_inputs = tf.stack(moe_inputs,axis=1)
        weight2d = tf.get_variable("ensemble_weight2d",
                                   shape=[num_extend, lstm_size, vocab_size],
                                   regularizer=slim.l2_regularizer(1.0e-8))
        weight = tf.nn.softmax(tf.einsum("aij,ijk->aik", moe_inputs, weight2d), dim=1)
        result = {}
        result["prediction_frames"] = tf.reshape(final_probilities,[-1,vocab_size])
        result["predictions"] = tf.reduce_sum(final_probilities*weight,axis=1)
        return result

class LstmMultiscale2Model(models.BaseModel):

    def cnn(self,
            model_input,
            l2_penalty=1e-8,
            num_filters=[1024,1024,1024],
            filter_sizes=[1,2,3],
            sub_scope="",
            **unused_params):
        max_frames = model_input.get_shape().as_list()[1]
        num_features = model_input.get_shape().as_list()[2]

        shift_inputs = []
        for i in range(max(filter_sizes)):
            if i == 0:
                shift_inputs.append(model_input)
            else:
                shift_inputs.append(tf.pad(model_input, paddings=[[0,0],[i,0],[0,0]])[:,:max_frames,:])

        cnn_outputs = []
        for nf, fs in zip(num_filters, filter_sizes):
            sub_input = tf.concat(shift_inputs[:fs], axis=2)
            sub_filter = tf.get_variable(sub_scope+"cnn-filter-len%d"%fs,
                                         shape=[num_features*fs, nf], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                         regularizer=tf.contrib.layers.l2_regularizer(l2_penalty))
            cnn_outputs.append(tf.einsum("ijk,kl->ijl", sub_input, sub_filter))

        cnn_output = tf.concat(cnn_outputs, axis=2)
        cnn_output = slim.batch_norm(
            cnn_output,
            center=True,
            scale=True,
            is_training=FLAGS.train,
            scope=sub_scope+"cluster_bn")
        return cnn_output, max_frames

    def sub_moe(self,model_input,
                vocab_size,
                num_mixtures=None,
                l2_penalty=1e-8,
                scopename="",
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

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates"+scopename)
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts"+scopename)

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures


        final_probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)

        final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                         [-1, vocab_size])

        return final_probabilities


    def rnn_gate(self, model_input, lstm_size, num_frames, l2_penalty=1e-8, sub_scope="", **unused_params):
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
        ## Batch normalize the input

        max_frames = model_input.get_shape().as_list()[1]
        emb_dim = model_input.get_shape().as_list()[2]
        hidden_dim = lstm_size

        with tf.variable_scope(sub_scope+'lstm_forward'):
            g_recurrent_unit_forward = LstmGateModel().create_recurrent_unit(emb_dim,hidden_dim,l2_penalty)

        h0 = tf.zeros([tf.shape(model_input)[0], hidden_dim])
        h0 = tf.stack([h0, h0])

        outputs_state = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=max_frames,
            dynamic_size=False, infer_shape=True)

        def _pretrain_forward(i,h_tm1,s_predictions):
            x_t = model_input[:,i,:]
            gate, h_t = g_recurrent_unit_forward(x_t, h_tm1)
            hidden_state, c_prev = tf.unstack(h_t)
            s_predictions = s_predictions.write(i,c_prev)
            return i + 1, h_t, s_predictions

        _, _, state_outputs = control_flow_ops.while_loop(
            cond=lambda i, _1, _2: i < max_frames,
            body=_pretrain_forward,
            loop_vars=(tf.constant(0, dtype=tf.int32),h0,outputs_state),
            swap_memory=True)

        state_outputs = state_outputs.stack()
        batch_size = tf.shape(model_input)[0]
        index_1 = tf.range(0, batch_size) * max_frames + (num_frames - 1)
        state_outputs = tf.transpose(state_outputs, [1, 0, 2])
        state_outputs = tf.gather(tf.reshape(state_outputs, [-1, hidden_dim]), index_1)

        return state_outputs

    def rnn_standard(self, model_input, lstm_size, num_frames,sub_scope="", **unused_params):
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
        ## Batch normalize the input
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=True)
                for _ in range(1)
                ],
            state_is_tuple=True)
        with tf.variable_scope("RNN-"+sub_scope):
            outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                               sequence_length=num_frames,
                                               swap_memory=True,
                                               dtype=tf.float32)

        state_out = tf.concat(map(lambda x: x.c, state), axis=1)

        return state_out


    def create_model(self, model_input, vocab_size, num_frames, l2_penalty=1e-8, **unused_params):

        num_extend = FLAGS.moe_num_extend
        num_layers = num_extend
        lstm_size = FLAGS.lstm_cells
        pool_size = 2
        cnn_input = model_input
        num_filters = [256, 256, 512]
        filter_sizes = [1, 2, 3]
        features_size = sum(num_filters)
        final_probilities = []
        moe_inputs = []
        for layer in range(num_layers):
            cnn_output, num_t = self.cnn(cnn_input, num_filters=num_filters, filter_sizes=filter_sizes, sub_scope="cnn%d"%(layer+1))
            cnn_output = tf.nn.relu(cnn_output)
            cnn_multiscale = self.rnn_gate(cnn_output, lstm_size, num_frames, sub_scope="rnn%d"%(layer+1))
            moe_inputs.append(cnn_multiscale)
            final_probility = self.sub_moe(cnn_multiscale, vocab_size, scopename="moe%d"%(layer+1))
            final_probilities.append(final_probility)
            num_t = pool_size*(num_t//pool_size)
            cnn_output = tf.reshape(cnn_output[:,:num_t,:],[-1,num_t//pool_size,pool_size,features_size])
            cnn_input = tf.reduce_max(cnn_output, axis=2)
            num_frames = tf.maximum(num_frames//pool_size,1)

        final_probilities = tf.stack(final_probilities, axis=1)
        moe_inputs = tf.stack(moe_inputs, axis=1)
        weight2d = tf.get_variable("ensemble_weight2d",
                                   shape=[num_extend, features_size, vocab_size],
                                   regularizer=slim.l2_regularizer(1.0e-8))
        weight = tf.nn.softmax(tf.einsum("aij,ijk->aik", moe_inputs, weight2d), dim=1)
        result = {}
        result["prediction_frames"] = tf.reshape(final_probilities,[-1, vocab_size])
        result["predictions"] = tf.reduce_mean(final_probilities, axis=1)
        return result

class LstmMultiscale3Model(models.BaseModel):

    def cnn(self,
            model_input,
            l2_penalty=1e-8,
            num_filters=[1024,1024,1024],
            filter_sizes=[1,2,3],
            sub_scope="",
            **unused_params):
        max_frames = model_input.get_shape().as_list()[1]
        num_features = model_input.get_shape().as_list()[2]

        shift_inputs = []
        for i in range(max(filter_sizes)):
            if i == 0:
                shift_inputs.append(model_input)
            else:
                shift_inputs.append(tf.pad(model_input, paddings=[[0,0],[i,0],[0,0]])[:,:max_frames,:])

        cnn_outputs = []
        for nf, fs in zip(num_filters, filter_sizes):
            sub_input = tf.concat(shift_inputs[:fs], axis=2)
            sub_filter = tf.get_variable(sub_scope+"cnn-filter-len%d"%fs,
                                         shape=[num_features*fs, nf], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                         regularizer=tf.contrib.layers.l2_regularizer(l2_penalty))
            cnn_outputs.append(tf.einsum("ijk,kl->ijl", sub_input, sub_filter))

        cnn_output = tf.concat(cnn_outputs, axis=2)
        cnn_output = slim.batch_norm(
            cnn_output,
            center=True,
            scale=True,
            is_training=FLAGS.train,
            scope=sub_scope+"cluster_bn")
        return cnn_output, max_frames

    def sub_moe(self,model_input,
                vocab_size,
                num_mixtures=None,
                l2_penalty=1e-8,
                scopename="",
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

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates"+scopename)
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts"+scopename)

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures


        final_probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)

        final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                         [-1, vocab_size])

        return final_probabilities


    def rnn_glu(self, model_input, lstm_size, num_frames, l2_penalty=1e-8, sub_scope="", **unused_params):
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
        ## Batch normalize the input

        max_frames = model_input.get_shape().as_list()[1]
        emb_dim = model_input.get_shape().as_list()[2]
        hidden_dim = lstm_size

        with tf.variable_scope(sub_scope+'lstm_forward'):
            g_recurrent_unit_forward = LstmGlu2Model().create_recurrent_unit(emb_dim,hidden_dim,l2_penalty)

        h0 = tf.zeros([tf.shape(model_input)[0], hidden_dim])
        h0 = tf.stack([h0, h0])
        h1 = tf.zeros([tf.shape(model_input)[0], emb_dim])
        h1 = tf.stack([h1, h1])

        outputs_state = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=max_frames,
            dynamic_size=False, infer_shape=True)

        def _pretrain_forward(i, h_tm0, h_tm1,s_predictions):
            x_t = model_input[:,i,:]
            gate, h_t0, h_t1 = g_recurrent_unit_forward(x_t, h_tm0, h_tm1)
            hidden_state, c_prev = tf.unstack(h_t1)
            s_predictions = s_predictions.write(i,hidden_state)
            return i + 1, h_t0, h_t1, s_predictions

        _, _, _, state_outputs = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3: i < max_frames,
            body=_pretrain_forward,
            loop_vars=(tf.constant(0, dtype=tf.int32), h1, h0, outputs_state),
            swap_memory=True)

        state_outputs = state_outputs.stack()
        batch_size = tf.shape(model_input)[0]
        index_1 = tf.range(0, batch_size) * max_frames + (num_frames - 1)
        state_outputs = tf.transpose(state_outputs, [1, 0, 2])
        state_outputs = tf.gather(tf.reshape(state_outputs, [-1, hidden_dim]), index_1)

        return state_outputs

    def rnn_standard(self, model_input, lstm_size, num_frames,sub_scope="", **unused_params):
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
        ## Batch normalize the input
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=True)
                for _ in range(1)
                ],
            state_is_tuple=True)
        with tf.variable_scope("RNN-"+sub_scope):
            outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                               sequence_length=num_frames,
                                               swap_memory=True,
                                               dtype=tf.float32)

        state_out = tf.concat(map(lambda x: x.c, state), axis=1)

        return state_out


    def create_model(self, model_input, vocab_size, num_frames, l2_penalty=1e-8, **unused_params):

        num_extend = FLAGS.moe_num_extend
        num_layers = num_extend
        lstm_size = FLAGS.lstm_cells
        pool_size = 2
        cnn_input = model_input
        num_filters = [256, 256, 512]
        filter_sizes = [1, 2, 3]
        features_size = sum(num_filters)
        final_probilities = []
        moe_inputs = []
        for layer in range(num_layers):
            cnn_output, num_t = self.cnn(cnn_input, num_filters=num_filters, filter_sizes=filter_sizes, sub_scope="cnn%d"%(layer+1))
            cnn_output = tf.nn.relu(cnn_output)
            cnn_multiscale = self.rnn_glu(cnn_output, lstm_size, num_frames, sub_scope="rnn%d"%(layer+1))
            moe_inputs.append(cnn_multiscale)
            final_probility = self.sub_moe(cnn_multiscale, vocab_size, scopename="moe%d"%(layer+1))
            final_probilities.append(final_probility)
            num_t = pool_size*(num_t//pool_size)
            cnn_output = tf.reshape(cnn_output[:,:num_t,:],[-1,num_t//pool_size,pool_size,features_size])
            cnn_input = tf.reduce_max(cnn_output, axis=2)
            num_frames = tf.maximum(num_frames//pool_size,1)

        final_probilities = tf.stack(final_probilities, axis=1)
        moe_inputs = tf.stack(moe_inputs, axis=1)
        weight2d = tf.get_variable("ensemble_weight2d",
                                   shape=[num_extend, features_size, vocab_size],
                                   regularizer=slim.l2_regularizer(1.0e-8))
        weight = tf.nn.softmax(tf.einsum("aij,ijk->aik", moe_inputs, weight2d), dim=1)
        result = {}
        result["prediction_frames"] = tf.reshape(final_probilities,[-1, vocab_size])
        result["predictions"] = tf.reduce_mean(final_probilities, axis=1)
        return result

class LstmMultiscaleRebuildModel(models.BaseModel):

    def create_model(self, model_input, vocab_size, num_frames, l2_penalty=1e-8, **unused_params):

        num_extend = FLAGS.moe_num_extend
        num_layers = num_extend
        lstm_size = FLAGS.lstm_cells
        pool_size=2
        cnn_input = model_input
        num_filters=[256,256,512]
        filter_sizes=[1,2,3]
        features_size = sum(num_filters)
        final_probilities = []
        moe_inputs = []

        for layer in range(num_layers):
            cnn_output, num_t = LstmMultiscaleModel().cnn(cnn_input, num_filters=num_filters, filter_sizes=filter_sizes, sub_scope="cnn%d"%(layer+1))
            cnn_output = tf.nn.relu(cnn_output)
            cnn_multiscale = LstmMultiscaleModel().rnn(cnn_output,lstm_size, num_frames,sub_scope="rnn%d"%(layer+1))
            moe_inputs.append(cnn_multiscale)
            final_probility = LstmMultiscaleModel().sub_moe(cnn_multiscale,vocab_size,scopename="moe%d"%(layer+1))
            final_probilities.append(final_probility)
            num_t = pool_size*(num_t//pool_size)
            cnn_output = tf.reshape(cnn_output[:,:num_t,:],[-1,num_t//pool_size,pool_size,features_size])
            cnn_input = tf.reduce_max(cnn_output, axis=2)
            num_frames = tf.maximum(num_frames//pool_size,1)

        final_probilities = tf.stack(final_probilities, axis=1)
        moe_inputs = tf.stack(moe_inputs, axis=1)
        weight2d = tf.get_variable("ensemble_weight2d",
                                   shape=[num_extend, features_size, vocab_size],
                                   regularizer=slim.l2_regularizer(1.0e-8))
        weight = tf.nn.softmax(tf.einsum("aij,ijk->aik", tf.stop_gradient(moe_inputs), weight2d), dim=1)
        result = {}
        result["predictions"] = tf.reduce_sum(tf.stop_gradient(final_probilities)*weight, axis=1)
        return result

class LstmLayerModel(models.BaseModel):

    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
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
        unit_layer_1 = FLAGS.lstm_length
        unit_layer_2 = shape[1]//FLAGS.lstm_length
        model_input = model_input[:,:unit_layer_1*unit_layer_2,:]

        model_input_1 = tf.reshape(model_input,[-1,unit_layer_1,shape[2]])
        ## Batch normalize the input
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=True)
                for _ in range(1)
                ],
            state_is_tuple=True)

        with tf.variable_scope("RNN-1"):
            outputs_1, state_1 = tf.nn.dynamic_rnn(stacked_lstm, model_input_1,
                                               sequence_length=None,
                                               swap_memory=True,
                                               dtype=tf.float32)
        state_c = tf.concat(map(lambda x: x.c, state_1), axis=1)
        model_input_2 = tf.reshape(state_c,[-1,unit_layer_2,lstm_size])
        num_frames = num_frames//unit_layer_1
        with tf.variable_scope("RNN-2"):
            outputs_2, state_2 = tf.nn.dynamic_rnn(stacked_lstm, model_input_2,
                                               sequence_length=num_frames,
                                               swap_memory=True,
                                               dtype=tf.float32)

        state_out = tf.concat(map(lambda x: x.c, state_2), axis=1)
        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=state_out,
            vocab_size=vocab_size,
            **unused_params)

class LstmDivideModel(models.BaseModel):

    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
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
        num_extend = FLAGS.moe_num_extend - 1
        shape = model_input.get_shape().as_list()
        full_num_frames = num_frames
        num_frames = tf.maximum(num_frames//num_extend,1)
        length = shape[1]//num_extend

        divide_inputs = []
        divide_inputs.append(model_input)
        for i in range(num_extend):
            begin_frames = tf.reshape(num_frames*i,[-1,1])
            frames_index = tf.reshape(tf.range(length),[1,length])
            frames_index = begin_frames+frames_index
            batch_size = tf.shape(model_input)[0]
            batch_index = tf.tile(
                tf.expand_dims(tf.range(batch_size), 1), [1, length])
            index = tf.stack([batch_index, tf.cast(frames_index,dtype=tf.int32)], 2)
            divide_input = tf.gather_nd(model_input, index)
            divide_input = tf.pad(divide_input, paddings=[[0,0],[0,shape[1]-length],[0,0]])
            divide_inputs.append(divide_input)

        divide_inputs = tf.reshape(tf.stack(divide_inputs,axis=1),[-1,shape[1],shape[2]])

        ## Batch normalize the input
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=True)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=True)
        num_frames = tf.tile(tf.reshape(num_frames,[-1,1]),[1,num_extend])
        num_frames = tf.concat((tf.reshape(full_num_frames,[-1,1]),num_frames),axis=1)
        num_frames = tf.reshape(num_frames,[-1])
        with tf.variable_scope("RNN"):
            outputs, state = tf.nn.dynamic_rnn(stacked_lstm, divide_inputs,
                                                   sequence_length=num_frames,
                                                   swap_memory=True,
                                                   dtype=tf.float32)
        state_c = tf.concat(map(lambda x: x.c, state), axis=1)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        result = aggregated_model().create_model(
            model_input=state_c,
            vocab_size=vocab_size,
            **unused_params)

        final_probilities = result["predictions"]
        result["prediction_frames"] = final_probilities
        result["predictions"] = tf.reduce_mean(tf.reshape(final_probilities,[-1,num_extend+1,vocab_size]),axis=1)
        return result

class LstmDivideRebuildModel(models.BaseModel):

    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
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

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        result = aggregated_model().create_model(
            model_input=state_c,
            vocab_size=vocab_size,
            **unused_params)

        return result


class LstmDivideRebuild2Model(models.BaseModel):

    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
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
        num_extend = FLAGS.moe_num_extend - 1
        shape = model_input.get_shape().as_list()
        full_num_frames = num_frames
        num_frames = tf.maximum(num_frames//num_extend,1)
        length = shape[1]//num_extend

        divide_inputs = []
        divide_inputs.append(model_input)
        for i in range(num_extend):
            begin_frames = tf.reshape(num_frames*i,[-1,1])
            frames_index = tf.reshape(tf.range(length),[1,length])
            frames_index = begin_frames+frames_index
            batch_size = tf.shape(model_input)[0]
            batch_index = tf.tile(
                tf.expand_dims(tf.range(batch_size), 1), [1, length])
            index = tf.stack([batch_index, tf.cast(frames_index,dtype=tf.int32)], 2)
            divide_input = tf.gather_nd(model_input, index)
            divide_input = tf.pad(divide_input, paddings=[[0,0],[0,shape[1]-length],[0,0]])
            divide_inputs.append(divide_input)

        divide_inputs = tf.reshape(tf.stack(divide_inputs,axis=1),[-1,shape[1],shape[2]])

        ## Batch normalize the input
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=True)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=True)
        num_frames = tf.tile(tf.reshape(num_frames,[-1,1]),[1,num_extend])
        num_frames = tf.concat((tf.reshape(full_num_frames,[-1,1]),num_frames),axis=1)
        num_frames = tf.reshape(num_frames,[-1])
        with tf.variable_scope("RNN"):
            outputs, state = tf.nn.dynamic_rnn(stacked_lstm, divide_inputs,
                                               sequence_length=num_frames,
                                               swap_memory=True,
                                               dtype=tf.float32)
        state_c = tf.concat(map(lambda x: x.c, state), axis=1)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        result = aggregated_model().create_model(
            model_input=state_c,
            vocab_size=vocab_size,
            **unused_params)
        features_size = state_c.get_shape().as_list()[1]
        final_probilities = result["predictions"]
        final_probilities_divide = tf.reshape(final_probilities,[-1,num_extend,vocab_size])
        moe_inputs = tf.reshape(state_c,[-1, num_extend, features_size])
        weight2d = tf.get_variable("notrestore_ensemble_weight2d",
                                   shape=[features_size, vocab_size],
                                   regularizer=slim.l2_regularizer(1.0e-8))
        weight = tf.nn.softmax(tf.einsum("aij,jk->aik", tf.stop_gradient(moe_inputs), weight2d), dim=1)
        result = {}
        result["predictions"] = tf.reduce_sum(tf.stop_gradient(final_probilities_divide)*weight,axis=1)
        return result

class LstmResidualModel(models.BaseModel):



    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
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
        shortcut_interval = FLAGS.lstm_interval
        shape = model_input.get_shape().as_list()

        ## Batch normalize the input
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=True)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=True)

        with tf.variable_scope("RNN"):
            outputs, state = rnn_residual.dynamic_rnn(stacked_lstm, model_input,shortcut_interval,
                                               sequence_length=num_frames,
                                               swap_memory=True,
                                               dtype=tf.float32)

        state_c = tf.concat(map(lambda x: x.c, state), axis=1)
        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=state_c,
            vocab_size=vocab_size,
            **unused_params)

class LstmNoiseModel(models.BaseModel):

    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
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

        if FLAGS.train=="train":
            noise = tf.random_normal(shape=tf.shape(model_input), mean=0.0, stddev=FLAGS.noise_std, dtype=tf.float32)
            model_input = tf.nn.l2_normalize(model_input+noise, 2)

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
        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=state_c,
            vocab_size=vocab_size,
            **unused_params)

class LstmConditionModel(models.BaseModel):

    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
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
        batch_size = tf.shape(model_input)[0]
        condition_input = tf.tile(tf.reshape(tf.diag(tf.ones(shape[1])),[1,shape[1],shape[1]]),[batch_size,1,1])
        model_input = tf.concat((model_input,condition_input),axis=2)
        model_input = tf.nn.l2_normalize(model_input, 2)
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
        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=state_c,
            vocab_size=vocab_size,
            **unused_params)

class LstmRandomModel(models.BaseModel):

    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
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

        if FLAGS.train=="train":
            shape = model_input.get_shape().as_list()
            frames_sum = tf.reduce_sum(tf.abs(model_input),axis=2)
            frames_index = tf.reshape(tf.range(shape[1],dtype=tf.float32),[1,shape[1]])
            frames_index = tf.ones(tf.shape(frames_sum))*frames_index
            frames_index = tf.transpose(tf.random_shuffle(tf.transpose(frames_index)))
            frames_index = frames_index[:,0:shape[1]//2]
            frames_index = tf.negative(frames_index)
            frames_index, indexes = tf.nn.top_k(frames_index, sorted=True, k=shape[1]//2)
            frames_index = tf.negative(frames_index)
            frames_valid = tf.cast(tf.reshape(num_frames,[-1,1]),dtype=tf.float32)
            frames_true = tf.ones(tf.shape(frames_index))
            frames_false = tf.zeros(tf.shape(frames_index))
            frames_bool = tf.where(tf.less(frames_index, frames_valid), frames_true, frames_false)
            num_frames = tf.reduce_sum(frames_bool,axis=1)

            batch_size = tf.shape(model_input)[0]
            batch_index = tf.tile(
                tf.expand_dims(tf.range(batch_size), 1), [1, shape[1]//2])
            index = tf.stack([batch_index, tf.cast(frames_index,dtype=tf.int32)], 2)
            model_input = tf.gather_nd(model_input, index)

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
        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=state_c,
            vocab_size=vocab_size,
            **unused_params)

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
        aggregated_model = getattr(video_level_models,"MoeMix4Model")
        result = aggregated_model().create_model(model_input=video_input,vocab_size=vocab_size,
                                                 **unused_params)
        #result["bottleneck"] = state_c
        mse_loss = tf.square(de_outputs-model_input)
        mse_loss = tf.reduce_mean(tf.reduce_sum(mse_loss, 2))
        result["loss"] = mse_loss
        return result

class LstmSoftmaxModel(models.BaseModel):

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
        lstm_size = FLAGS.lstm_cells
        number_of_layers = FLAGS.lstm_layers
        shape = model_input.get_shape().as_list()
        num_mixtures = FLAGS.moe_num_mixtures

        frames_sum = tf.reduce_sum(tf.abs(model_input),axis=2)
        frames_true = tf.ones(tf.shape(frames_sum))
        frames_false = tf.zeros(tf.shape(frames_sum))
        frames_bool = tf.reshape(tf.where(tf.greater(frames_sum, frames_false), frames_true, frames_false),[-1,shape[1],1])

        ## Batch normalize the input
        """
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
                                               dtype=tf.float32)"""
        lstm_size = shape[2]
        outputs = model_input
        softmax_input = tf.reduce_mean(tf.reshape(outputs,[-1,FLAGS.stride_size,lstm_size]),axis=1)
        frames_bool = frames_bool[:,0:shape[1]:FLAGS.stride_size,:]
        expert_activations = slim.fully_connected(
            softmax_input,
            num_mixtures*vocab_size,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="expert_activations")
        gate_activations = slim.fully_connected(
            softmax_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gate_activations")
        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        probabilities_by_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)


        gate = slim.fully_connected(
            softmax_input, vocab_size, activation_fn=None,
            weights_regularizer=slim.l2_regularizer(1e-8),
            scope="gates")

        gate = tf.nn.softmax(tf.reshape(gate,[-1, shape[1]//FLAGS.stride_size,vocab_size]),dim=1)
        gate = gate*frames_bool
        gate = gate/tf.reduce_sum(gate, axis=1, keep_dims=True)

        output = tf.reshape(probabilities_by_batch,[-1, shape[1]//FLAGS.stride_size,vocab_size])
        perdiction_frames = output
        final_probabilities = tf.reduce_sum(perdiction_frames*gate,axis=1)
        return {"predictions": final_probabilities}

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

class CnnGluModel(models.BaseModel):
    def cnn(self,
            model_input,
            l2_penalty=1e-8,
            num_filters=[256, 256, 256],
            filter_sizes=[1,2,3],
            sub_scope="",
            **unused_params):
        max_frames = model_input.get_shape().as_list()[1]
        num_features = model_input.get_shape().as_list()[2]

        shift_inputs = []
        for i in range(max(filter_sizes)):
            if i == 0:
                shift_inputs.append(model_input)
            else:
                shift_inputs.append(tf.pad(model_input, paddings=[[0,0],[i,0],[0,0]])[:,:max_frames,:])

        cnn_outputs = []
        cnn_sigmoids = []
        for nf, fs in zip(num_filters, filter_sizes):
            sub_input = tf.concat(shift_inputs[:fs], axis=2)
            sub_filter = tf.get_variable(sub_scope+"cnn-filter-len%d"%fs,
                                         shape=[num_features*fs, nf], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                         regularizer=tf.contrib.layers.l2_regularizer(l2_penalty))
            cnn_outputs.append(tf.einsum("ijk,kl->ijl", sub_input, sub_filter))
            sigmoid_filter = tf.get_variable(sub_scope+"sigmoid-filter-len%d"%fs,
                                         shape=[num_features*fs, nf], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                         regularizer=tf.contrib.layers.l2_regularizer(l2_penalty))
            cnn_sigmoids.append(tf.nn.sigmoid(tf.einsum("ijk,kl->ijl", sub_input, sigmoid_filter)))

        cnn_output = tf.concat(cnn_outputs, axis=2)
        cnn_sigmoid = tf.concat(cnn_sigmoids, axis=2)
        if cnn_output.get_shape().as_list()[2]==model_input.get_shape().as_list()[2]:
            cnn_output = cnn_output*cnn_sigmoid + model_input
        else:
            cnn_output = cnn_output*cnn_sigmoid
        cnn_output = slim.batch_norm(
            cnn_output,
            center=True,
            scale=True,
            is_training=FLAGS.train,
            scope=sub_scope+"cluster_bn")
        return cnn_output, max_frames

    def kmax(self,
             model_input,
             l2_penalty=1e-8,
             num_filter=1024,
             filter_size=8,
             sub_scope="",
             **unused_params):
        max_frames = model_input.get_shape().as_list()[1]
        num_features = model_input.get_shape().as_list()[2]
        num_filters = [num_filter for _ in range(filter_size)]
        filter_sizes = [i*2+1 for i in range(filter_size)]

        shift_inputs = []
        for i in range(max(filter_sizes)):
            if i == 0:
                shift_inputs.append(model_input)
            else:
                shift_inputs.append(tf.pad(model_input, paddings=[[0,0],[i,0],[0,0]])[:,:max_frames,:])
        cnn_outputs = []
        for nf, fs in zip(num_filters, filter_sizes):
            sub_input = tf.concat(shift_inputs[:fs], axis=2)
            sub_filter = tf.get_variable(sub_scope+"cnn-filter-len%d"%fs,
                                         shape=[num_features*fs, nf], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                         regularizer=tf.contrib.layers.l2_regularizer(l2_penalty))
            cnn_outputs.append(tf.einsum("ijk,kl->ijl", sub_input, sub_filter))

        cnn_output = tf.stack(cnn_outputs, axis=2)
        cnn_output = tf.reduce_max(cnn_output, axis=1)

        return cnn_output, max_frames

    def sub_moe(self,
                model_input,
                vocab_size,
                num_mixtures=None,
                l2_penalty=1e-8,
                scopename="",
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

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates"+scopename)
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts"+scopename)

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures


        final_probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)

        final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                         [-1, vocab_size])

        return final_probabilities

    def create_model(self, model_input, vocab_size, num_frames, l2_penalty=1e-8, **unused_params):

        num_extend = FLAGS.moe_num_extend
        num_layers = 10
        pool_size=2
        cnn_input = model_input
        num_filters=[256,256,512]
        filter_sizes=[1,2,3]
        features_size = sum(num_filters)

        for layer in range(num_layers):
            cnn_output, num_t = self.cnn(cnn_input, num_filters=num_filters, filter_sizes=filter_sizes, sub_scope="cnn%d"%(layer+1))
            if layer < 3:
                num_t = pool_size*(num_t//pool_size)
                cnn_output = tf.reshape(cnn_output[:,:num_t,:],[-1,num_t//pool_size,pool_size,features_size])
                cnn_input = tf.reduce_max(cnn_output, axis=2)
            else:
                cnn_input = cnn_output

        cnn_output, num_t = self.kmax(cnn_input, num_filters=features_size, filter_sizes=num_extend, sub_scope="kmax")
        cnn_input = tf.reshape(cnn_output,[-1,features_size])
        final_probilities = self.sub_moe(cnn_input,vocab_size)
        final_probilities = tf.reshape(final_probilities,[-1,num_extend,vocab_size])
        weight2d = tf.get_variable("ensemble_weight2d",
                                   shape=[num_extend, features_size, vocab_size],
                                   regularizer=slim.l2_regularizer(1.0e-8))
        weight = tf.nn.softmax(tf.einsum("aij,ijk->aik", cnn_output, weight2d), dim=1)
        result = {}
        result["predictions"] = tf.reduce_sum(final_probilities*weight,axis=1)
        return result

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

class CnnKmaxModel(models.BaseModel):

    def cnn(self,
            model_input,
            l2_penalty=1e-8,
            num_filters = [1024, 1024, 1024],
            filter_sizes = [1,2,3],
            sub_scope="",
            **unused_params):
        max_frames = model_input.get_shape().as_list()[1]
        num_features = model_input.get_shape().as_list()[2]

        shift_inputs = []
        for i in range(max(filter_sizes)):
            if i == 0:
                shift_inputs.append(model_input)
            else:
                shift_inputs.append(tf.pad(model_input, paddings=[[0,0],[i,0],[0,0]])[:,:max_frames,:])

        cnn_outputs = []
        for nf, fs in zip(num_filters, filter_sizes):
            sub_input = tf.concat(shift_inputs[:fs], axis=2)
            sub_filter = tf.get_variable(sub_scope+"cnn-filter-len%d"%fs,
                                         shape=[num_features*fs, nf], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                         regularizer=tf.contrib.layers.l2_regularizer(l2_penalty))
            cnn_outputs.append(tf.einsum("ijk,kl->ijl", sub_input, sub_filter))

        cnn_output = tf.concat(cnn_outputs, axis=2)
        cnn_output = slim.batch_norm(
            cnn_output,
            center=True,
            scale=True,
            is_training=FLAGS.train,
            scope=sub_scope+"cluster_bn")
        return cnn_output, max_frames


    def sub_moe(self,
                model_input,
                vocab_size,
                num_mixtures=None,
                l2_penalty=1e-8,
                scopename="",
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

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates"+scopename)
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts"+scopename)

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures


        final_probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)

        final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                         [-1, vocab_size])

        return final_probabilities

    def create_model(self, model_input, vocab_size, num_frames, l2_penalty=1e-8, **unused_params):

        num_extend = FLAGS.moe_num_extend
        num_layers = num_extend
        pool_size=2
        cnn_input = model_input
        num_filters=[256,256,512]
        filter_sizes=[1,2,3]
        features_size = sum(num_filters)
        final_probilities = []
        moe_inputs = []
        for layer in range(num_layers):
            cnn_output, num_t = self.cnn(cnn_input, num_filters=num_filters, filter_sizes=filter_sizes, sub_scope="cnn%d"%(layer+1))
            cnn_output = tf.nn.relu(cnn_output)
            cnn_multiscale = tf.reduce_max(cnn_output,axis=1)
            moe_inputs.append(cnn_multiscale)
            final_probility = self.sub_moe(cnn_multiscale,vocab_size,scopename="moe%d"%(layer+1))
            final_probilities.append(final_probility)
            num_t = pool_size*(num_t//pool_size)
            cnn_output = tf.reshape(cnn_output[:,:num_t,:],[-1,num_t//pool_size,pool_size,features_size])
            cnn_input = tf.reduce_max(cnn_output, axis=2)

        final_probilities = tf.stack(final_probilities,axis=1)
        moe_inputs = tf.stack(moe_inputs,axis=1)
        weight2d = tf.get_variable("ensemble_weight2d",
                                   shape=[num_extend, features_size, vocab_size],
                                   regularizer=slim.l2_regularizer(1.0e-8))
        weight = tf.nn.softmax(tf.einsum("aij,ijk->aik", moe_inputs, weight2d), dim=1)
        result = {}
        result["prediction_frames"] = tf.reshape(final_probilities,[-1,vocab_size])
        result["predictions"] = tf.reduce_sum(final_probilities*weight,axis=1)
        return result

class CnnKmaxRebuildModel(models.BaseModel):

    def create_model(self, model_input, vocab_size, num_frames, l2_penalty=1e-8, **unused_params):
        num_extend = FLAGS.moe_num_extend
        num_layers = num_extend
        pool_size=2
        cnn_input = model_input
        num_filters=[256,256,512]
        filter_sizes=[1,2,3]
        features_size = sum(num_filters)
        final_probilities = []
        moe_inputs = []
        for layer in range(num_layers):
            cnn_output, num_t = CnnKmaxModel().cnn(cnn_input, num_filters=num_filters, filter_sizes=filter_sizes, sub_scope="cnn%d"%(layer+1), l2_penalty=0.0)
            cnn_output = tf.nn.relu(cnn_output)
            cnn_multiscale = tf.reduce_max(cnn_output,axis=1)
            moe_inputs.append(cnn_multiscale)
            final_probility = CnnKmaxModel().sub_moe(cnn_multiscale,vocab_size,scopename="moe%d"%(layer+1), l2_penalty=0.0)
            final_probilities.append(final_probility)
            num_t = pool_size*(num_t//pool_size)
            cnn_output = tf.reshape(cnn_output[:,:num_t,:],[-1,num_t//pool_size,pool_size,features_size])
            cnn_input = tf.reduce_max(cnn_output, axis=2)

        final_probilities = tf.stack(final_probilities,axis=1)
        moe_inputs = tf.stack(moe_inputs,axis=1)
        weight2d = tf.get_variable("ensemble_weight2d",
                                   shape=[num_extend, features_size, vocab_size],
                                   regularizer=slim.l2_regularizer(1.0e-8))
        weight = tf.nn.softmax(tf.einsum("aij,ijk->aik", tf.stop_gradient(moe_inputs), weight2d), dim=1)
        result = {}
        result["predictions"] = tf.reduce_sum(tf.stop_gradient(final_probilities)*weight, axis=1)
        return result

class CnnWholeModel(models.BaseModel):

    def cnn(self,
            model_input,
            l2_penalty=1e-8,
            num_filters=[4,512],
            sub_scope="",
            **unused_params):
        max_frames = model_input.get_shape().as_list()[1]
        num_features = model_input.get_shape().as_list()[2]
        sub_filter_1 = tf.get_variable(sub_scope+"cnn-filter-1",
                                     shape=[max_frames, num_filters[0]], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                     regularizer=tf.contrib.layers.l2_regularizer(l2_penalty))
        sub_bias_1 = tf.get_variable(sub_scope+"cnn-bias-1",
                                       shape=[num_filters[0]], dtype=tf.float32,
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                       regularizer=tf.contrib.layers.l2_regularizer(l2_penalty))
        cnn_outputs_1 = tf.einsum("ijk,jl->ikl", model_input, sub_filter_1) + sub_bias_1

        cnn_outputs_1 = slim.batch_norm(
            cnn_outputs_1,
            center=True,
            scale=True,
            is_training=FLAGS.train,
            scope=sub_scope+"cluster_bn_1")

        cnn_outputs_1 = tf.nn.relu(cnn_outputs_1)

        sub_filter_2 = tf.get_variable(sub_scope+"cnn-filter-2",
                                       shape=[num_features, num_filters[1]], dtype=tf.float32,
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                       regularizer=tf.contrib.layers.l2_regularizer(l2_penalty))
        sub_bias_2 = tf.get_variable(sub_scope+"cnn-bias-2",
                                     shape=[num_filters[1]], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                     regularizer=tf.contrib.layers.l2_regularizer(l2_penalty))
        cnn_outputs_2 = tf.einsum("ijk,jl->ikl", cnn_outputs_1, sub_filter_2) + sub_bias_2
        cnn_outputs_2 = tf.reshape(cnn_outputs_2,[-1,num_filters[0]*num_filters[1]])

        cnn_output = slim.batch_norm(
            cnn_outputs_2,
            center=True,
            scale=True,
            is_training=FLAGS.train,
            scope=sub_scope+"cluster_bn_2")

        return cnn_output


    def sub_moe(self,
                model_input,
                vocab_size,
                num_mixtures=None,
                l2_penalty=1e-8,
                scopename="",
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

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates"+scopename)
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts"+scopename)

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures


        final_probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)

        final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                         [-1, vocab_size])

        return final_probabilities

    def create_model(self, model_input, vocab_size, num_frames, l2_penalty=1e-8, **unused_params):
        cnn_output = self.cnn(model_input, sub_scope="cnn")
        final_probility = self.sub_moe(cnn_output,vocab_size,scopename="moe")
        result = {}
        result["predictions"] = final_probility
        return result

class CnnMultiscaleModel(models.BaseModel):

    def cnn(self,
            model_input,
            l2_penalty=1e-8,
            num_filters = [1024, 1024, 1024],
            filter_sizes = [1,2,3],
            sub_scope="",
            **unused_params):
        max_frames = model_input.get_shape().as_list()[1]
        num_features = model_input.get_shape().as_list()[2]

        shift_inputs = []
        for i in range(max(filter_sizes)):
            if i == 0:
                shift_inputs.append(model_input)
            else:
                shift_inputs.append(tf.pad(model_input, paddings=[[0,0],[i,0],[0,0]])[:,:max_frames,:])

        cnn_outputs = []
        for nf, fs in zip(num_filters, filter_sizes):
            sub_input = tf.concat(shift_inputs[:fs], axis=2)
            sub_filter = tf.get_variable(sub_scope+"cnn-filter-len%d"%fs,
                                         shape=[num_features*fs, nf], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                         regularizer=tf.contrib.layers.l2_regularizer(l2_penalty))
            sub_bias = tf.get_variable(sub_scope+"cnn-bias-len%d"%fs,
                                         shape=[nf], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                         regularizer=tf.contrib.layers.l2_regularizer(l2_penalty))
            cnn_outputs.append(tf.einsum("ijk,kl->ijl", sub_input, sub_filter) + sub_bias)

        cnn_output = tf.concat(cnn_outputs, axis=2)
        cnn_output = slim.batch_norm(
            cnn_output,
            center=True,
            scale=True,
            is_training=FLAGS.train,
            scope=sub_scope+"cluster_bn")
        return cnn_output, max_frames


    def sub_moe(self,
                model_input,
                vocab_size,
                num_mixtures=None,
                l2_penalty=1e-8,
                scopename="",
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

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates"+scopename)
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts"+scopename)

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures


        final_probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)

        final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                         [-1, vocab_size])

        return final_probabilities

    def create_model(self, model_input, vocab_size, num_frames, l2_penalty=1e-8, **unused_params):

        num_extend = FLAGS.moe_num_extend
        num_layers = num_extend
        pool_size=2
        cnn_input = model_input
        num_filters=[256,256,512]
        filter_sizes=[1,2,3]
        features_size = sum(num_filters)
        final_probilities = []
        moe_inputs = []
        for layer in range(num_layers):
            cnn_output, num_t = self.cnn(cnn_input, num_filters=num_filters, filter_sizes=filter_sizes, sub_scope="cnn%d"%(layer+1))
            cnn_output = tf.nn.relu(cnn_output)
            cnn_multiscale = tf.reduce_max(cnn_output,axis=1)
            moe_inputs.append(cnn_multiscale)
            final_probility = self.sub_moe(cnn_multiscale,vocab_size,scopename="moe%d"%(layer+1))
            final_probilities.append(final_probility)
            num_t = pool_size*(num_t//pool_size)
            cnn_output = tf.reshape(cnn_output[:,:num_t,:],[-1,num_t//pool_size,pool_size,features_size])
            cnn_input = tf.reduce_max(cnn_output, axis=2)

        final_probilities = tf.stack(final_probilities,axis=1)
        moe_inputs = tf.stack(moe_inputs,axis=1)
        weight2d = tf.get_variable("ensemble_weight2d",
                                   shape=[num_extend, features_size, vocab_size],
                                   regularizer=slim.l2_regularizer(1.0e-8))
        weight = tf.nn.softmax(tf.einsum("aij,ijk->aik", moe_inputs, weight2d), dim=1)
        result = {}
        result["prediction_frames"] = tf.reshape(final_probilities,[-1,vocab_size])
        result["predictions"] = tf.reduce_sum(final_probilities*weight,axis=1)
        return result

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


class LstmExtendModel(models.BaseModel):

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
            result = aggregated_model().create_model(
                model_input=pooled,
                vocab_size=vocab_size,
                **unused_params)
            result["attention_weight"] = tf.reshape(rnn_attention,[-1, shape[1]*num_extend])
            return result

class LstmExtendStateModel(models.BaseModel):
    def cnn(self,
            model_input,
            l2_penalty=1e-8,
            num_filters=[1024, 1024, 1024],
            filter_sizes=[1,2,3],
            sub_scope="",
            **unused_params):
        max_frames = model_input.get_shape().as_list()[1]
        num_features = model_input.get_shape().as_list()[2]

        shift_inputs = []
        for i in range(max(filter_sizes)):
            if i == 0:
                shift_inputs.append(model_input)
            else:
                shift_inputs.append(tf.pad(model_input, paddings=[[0,0],[i,0],[0,0]])[:,:max_frames,:])

        cnn_outputs = []
        for nf, fs in zip(num_filters, filter_sizes):
            sub_input = tf.concat(shift_inputs[:fs], axis=2)
            sub_filter = tf.get_variable(sub_scope+"cnn-filter-len%d"%fs,
                                         shape=[num_features*fs, nf], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                         regularizer=tf.contrib.layers.l2_regularizer(l2_penalty))
            cnn_outputs.append(tf.einsum("ijk,kl->ijl", sub_input, sub_filter))

        cnn_output = tf.concat(cnn_outputs, axis=2)
        return cnn_output, max_frames

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
        num_filters=[256,256,512]
        filter_sizes = [1,2,3]
        cnn_size = sum(num_filters)
        cnn_output, _ = self.cnn(model_input, num_filters=num_filters, filter_sizes=filter_sizes, sub_scope="cnn")
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
        final_output = tf.concat(map(lambda x: x.c, state), axis=1)
        final_output = tf.tile(tf.reshape(final_output,[-1,1,lstm_size*2]),[1,shape[1],1])
        W1 = tf.Variable(tf.truncated_normal([lstm_size*2,num_extend], stddev=0.1), name="W1")
        b1 = tf.Variable(tf.constant(0.1, shape=[num_extend]), name="b1")
        W2 = tf.Variable(tf.truncated_normal([cnn_size,num_extend], stddev=0.1), name="W2")
        b2 = tf.Variable(tf.constant(0.1, shape=[num_extend]), name="b2")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(W1))
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(b1))
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(W2))
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(b2))
        rnn_output = tf.reshape(final_output,[-1,lstm_size*2])
        rnn_input = tf.reshape(cnn_output,[-1,cnn_size])
        rnn_attention = tf.nn.softmax(tf.reshape(tf.nn.xw_plus_b(rnn_output,W1,b1)+tf.nn.xw_plus_b(rnn_input,W2,b2),[-1,shape[1],num_extend]),dim=1)*frames_bool
        rnn_attention = rnn_attention/tf.reduce_sum(rnn_attention, axis=1, keep_dims=True)
        rnn_attention = tf.transpose(rnn_attention,perm=[0,2,1])
        rnn_out = tf.einsum('aij,ajk->aik', rnn_attention, cnn_output)

        pooled = tf.reshape(rnn_out,[-1,cnn_size])

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        result = aggregated_model().create_model(
            model_input=pooled,
            vocab_size=vocab_size,
            **unused_params)
        result["attention_weight"] = tf.reshape(rnn_attention,[-1, shape[1]*num_extend])
        return result

class LstmExtendInputModel(models.BaseModel):

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
        rnn_out = tf.einsum('aij,ajk->aik', rnn_attention, model_input)

        pooled = tf.reshape(rnn_out,[-1,shape[2]])

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        result = aggregated_model().create_model(
            model_input=pooled,
            vocab_size=vocab_size,
            **unused_params)
        result["attention_weight"] = tf.reshape(rnn_attention,[-1, shape[1]*num_extend])
        return result

class LstmExtendCNNModel(models.BaseModel):

    def cnn(self,
            model_input,
            l2_penalty=1e-8,
            num_filters=[1024, 1024, 1024],
            filter_sizes=[1,2,3],
            sub_scope="",
            **unused_params):
        max_frames = model_input.get_shape().as_list()[1]
        num_features = model_input.get_shape().as_list()[2]

        shift_inputs = []
        for i in range(max(filter_sizes)):
            if i == 0:
                shift_inputs.append(model_input)
            else:
                shift_inputs.append(tf.pad(model_input, paddings=[[0,0],[i,0],[0,0]])[:,:max_frames,:])

        cnn_outputs = []
        for nf, fs in zip(num_filters, filter_sizes):
            sub_input = tf.concat(shift_inputs[:fs], axis=2)
            sub_filter = tf.get_variable(sub_scope+"cnn-filter-len%d"%fs,
                                         shape=[num_features*fs, nf], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                         regularizer=tf.contrib.layers.l2_regularizer(l2_penalty))
            cnn_outputs.append(tf.einsum("ijk,kl->ijl", sub_input, sub_filter))

        cnn_output = tf.concat(cnn_outputs, axis=2)
        return cnn_output, max_frames

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
        num_filters=[256,256,512]
        filter_sizes = [1,2,3]
        cnn_size = sum(num_filters)
        cnn_output, _ = self.cnn(model_input, num_filters=num_filters, filter_sizes=filter_sizes, sub_scope="cnn")
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
        W2 = tf.Variable(tf.truncated_normal([cnn_size,num_extend], stddev=0.1), name="W2")
        b2 = tf.Variable(tf.constant(0.1, shape=[num_extend]), name="b2")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(W1))
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(b1))
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(W2))
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(b2))
        rnn_output = tf.reshape(outputs,[-1,lstm_size])
        rnn_input = tf.reshape(cnn_output,[-1,cnn_size])
        rnn_attention = tf.nn.softmax(tf.reshape(tf.nn.xw_plus_b(rnn_output,W1,b1)+tf.nn.xw_plus_b(rnn_input,W2,b2),[-1,shape[1],num_extend]),dim=1)*frames_bool
        rnn_attention = rnn_attention/tf.reduce_sum(rnn_attention, axis=1, keep_dims=True)
        rnn_attention = tf.transpose(rnn_attention,perm=[0,2,1])

        rnn_out = tf.einsum('aij,ajk->aik', rnn_attention, cnn_output)

        pooled = tf.reshape(rnn_out,[-1,cnn_size])

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        result = aggregated_model().create_model(
            model_input=pooled,
            vocab_size=vocab_size,
            **unused_params)
        result["attention_weight"] = tf.reshape(rnn_attention,[-1, shape[1]*num_extend])
        return result

class LstmExtendOrthoModel(models.BaseModel):

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
        def calculate_ortho(W):
            index = tf.cast(tf.random_uniform([2])*num_extend, tf.int32)
            vec_1 = W[:, index[0]]
            vec_2 = W[:, index[1]]
            ortho = tf.square(tf.reduce_sum(vec_1*vec_2))/(tf.reduce_sum(tf.square(vec_1))*tf.reduce_sum(tf.square(vec_2)))
            return ortho
        W1 = tf.Variable(tf.truncated_normal([lstm_size,num_extend], stddev=0.1), name="W1")
        b1 = tf.Variable(tf.constant(0.1, shape=[num_extend]), name="b1")
        W2 = tf.Variable(tf.truncated_normal([shape[2],num_extend], stddev=0.1), name="W2")
        b2 = tf.Variable(tf.constant(0.1, shape=[num_extend]), name="b2")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(W1)+calculate_ortho(W1))
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(b1))
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(W2)+calculate_ortho(W1))
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

class LstmGateExtendModel(models.BaseModel):

    def rnn_gate(self, model_input, lstm_size, num_frames, l2_penalty=1e-8, sub_scope="", **unused_params):
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
        ## Batch normalize the input

        max_frames = model_input.get_shape().as_list()[1]
        emb_dim = model_input.get_shape().as_list()[2]
        hidden_dim = lstm_size

        with tf.variable_scope(sub_scope+'lstm_forward'):
            g_recurrent_unit_forward = LstmGateModel().create_recurrent_unit(emb_dim,hidden_dim,l2_penalty)

        h0 = tf.zeros([tf.shape(model_input)[0], hidden_dim])
        h0 = tf.stack([h0, h0])

        outputs_state = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=max_frames,
            dynamic_size=False, infer_shape=True)

        def _pretrain_forward(i,h_tm1,s_predictions):
            x_t = model_input[:,i,:]
            gate, h_t = g_recurrent_unit_forward(x_t, h_tm1)
            hidden_state, c_prev = tf.unstack(h_t)
            s_predictions = s_predictions.write(i,hidden_state)
            return i + 1, h_t, s_predictions

        _, _, state_outputs = control_flow_ops.while_loop(
            cond=lambda i, _1, _2: i < max_frames,
            body=_pretrain_forward,
            loop_vars=(tf.constant(0, dtype=tf.int32), h0, outputs_state),
            swap_memory=True)

        state_outputs = state_outputs.stack()
        state_outputs = tf.transpose(state_outputs, [1, 0, 2])

        return state_outputs

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
        lstm_size = FLAGS.lstm_cells
        num_extend = FLAGS.moe_num_extend
        shape = model_input.get_shape().as_list()
        frames_sum = tf.reduce_sum(tf.abs(model_input),axis=2)
        frames_true = tf.ones(tf.shape(frames_sum))
        frames_false = tf.zeros(tf.shape(frames_sum))
        frames_bool = tf.where(tf.greater(frames_sum, frames_false), frames_true, frames_false)
        frames_bool = tf.reshape(frames_bool,[-1,shape[1],1])

        with tf.variable_scope("RNN"):
            outputs = self.rnn_gate(model_input, lstm_size, num_frames)
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
        rnn_attention = rnn_attention/(tf.reduce_sum(rnn_attention, axis=1, keep_dims=True)+1e-6)
        rnn_attention = tf.transpose(rnn_attention,perm=[0,2,1])
        rnn_out = tf.einsum('aij,ajk->aik', rnn_attention, outputs)

        pooled = tf.reshape(rnn_out,[-1, lstm_size])

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=pooled,
            vocab_size=vocab_size,
            **unused_params)

class LstmGluExtendModel(models.BaseModel):

    def rnn_gate(self, model_input, lstm_size, num_frames, l2_penalty=1e-8, sub_scope="", **unused_params):
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
        ## Batch normalize the input

        max_frames = model_input.get_shape().as_list()[1]
        emb_dim = model_input.get_shape().as_list()[2]
        hidden_dim = lstm_size

        with tf.variable_scope(sub_scope+'lstm_forward'):
            g_recurrent_unit_forward = LstmGlu2Model().create_recurrent_unit(emb_dim, hidden_dim, l2_penalty)

        h0 = tf.zeros([tf.shape(model_input)[0], hidden_dim])
        h0 = tf.stack([h0, h0])
        h1 = tf.zeros([tf.shape(model_input)[0], emb_dim])
        h1 = tf.stack([h1, h1])

        outputs_state = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=max_frames,
            dynamic_size=False, infer_shape=True)

        def _pretrain_forward(i, h_tm0, h_tm1,s_predictions):
            x_t = model_input[:,i,:]
            gate, h_t0, h_t1 = g_recurrent_unit_forward(x_t, h_tm0, h_tm1)
            hidden_state, c_prev = tf.unstack(h_t1)
            s_predictions = s_predictions.write(i,hidden_state)
            return i + 1, h_t0, h_t1, s_predictions

        _, _, _, state_outputs = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3: i < max_frames,
            body=_pretrain_forward,
            loop_vars=(tf.constant(0, dtype=tf.int32), h1, h0, outputs_state),
            swap_memory=True)

        state_outputs = state_outputs.stack()
        state_outputs = tf.transpose(state_outputs, [1, 0, 2])

        return state_outputs

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
        num_extend = FLAGS.moe_num_extend
        shape = model_input.get_shape().as_list()
        frames_sum = tf.reduce_sum(tf.abs(model_input),axis=2)
        frames_true = tf.ones(tf.shape(frames_sum))
        frames_false = tf.zeros(tf.shape(frames_sum))
        frames_bool = tf.where(tf.greater(frames_sum, frames_false), frames_true, frames_false)
        frames_bool = tf.reshape(frames_bool,[-1,shape[1],1])

        with tf.variable_scope("RNN"):
            outputs = self.rnn_gate(model_input, lstm_size, num_frames)
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
        rnn_attention = rnn_attention/(tf.reduce_sum(rnn_attention, axis=1, keep_dims=True)+1e-6)
        rnn_attention = tf.transpose(rnn_attention,perm=[0, 2, 1])
        rnn_out = tf.einsum('aij,ajk->aik', rnn_attention, outputs)

        pooled = tf.reshape(rnn_out,[-1,lstm_size])

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=pooled,
            vocab_size=vocab_size,
            **unused_params)

class LstmExtendParallelModel(models.BaseModel):

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
        lstm_size_sys = FLAGS.lstm_cells
        number_of_layers = FLAGS.lstm_layers
        num_extend = FLAGS.moe_num_extend
        shape = model_input.get_shape().as_list()
        frames_sum = tf.reduce_sum(tf.abs(model_input),axis=2)
        frames_true = tf.ones(tf.shape(frames_sum))
        frames_false = tf.zeros(tf.shape(frames_sum))
        frames_bool = tf.where(tf.greater(frames_sum, frames_false), frames_true, frames_false)
        frames_bool = tf.reshape(frames_bool,[-1,shape[1],1])

        feature_sizes = FLAGS.feature_sizes
        feature_sizes = [int(feature_size) for feature_size in feature_sizes.split(',')]

        rgb_input = model_input[:,:,0:feature_sizes[0]]
        audio_input = model_input[:,:,feature_sizes[0]:]
        audio_input = audio_input/(tf.reduce_max(audio_input,axis=2,keep_dims=True)+1e-6)
        shape_i = rgb_input.get_shape().as_list()
        lstm_size = lstm_size_sys
        ## Batch normalize the input
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)

        with tf.variable_scope("RNN"):
            outputs, state = tf.nn.dynamic_rnn(stacked_lstm, rgb_input,
                                               sequence_length=num_frames,
                                               swap_memory=True,
                                               dtype=tf.float32)
            W1 = tf.Variable(tf.truncated_normal([lstm_size,num_extend], stddev=0.1), name="W1")
            b1 = tf.Variable(tf.constant(0.0, shape=[num_extend]), name="b1")
            W2 = tf.Variable(tf.truncated_normal([shape_i[2],num_extend], stddev=0.1), name="W2")
            b2 = tf.Variable(tf.constant(0.0, shape=[num_extend]), name="b2")
            tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(W1))
            tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(b1))
            tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(W2))
            tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(b2))
            rnn_output = tf.reshape(outputs,[-1,lstm_size])
            rnn_input = tf.reshape(rgb_input,[-1,shape_i[2]])
            rnn_attention = tf.nn.softmax(tf.reshape(tf.nn.xw_plus_b(rnn_output,W1,b1)+tf.nn.xw_plus_b(rnn_input,W2,b2),[-1,shape_i[1],num_extend]),dim=1)*frames_bool
            rnn_attention = rnn_attention/tf.reduce_sum(rnn_attention, axis=1, keep_dims=True)
            rnn_attention = tf.transpose(rnn_attention,perm=[0,2,1])
            rgb_out = tf.einsum('aij,ajk->aik', rnn_attention, outputs)
            audio_out = tf.einsum('aij,ajk->aik', rnn_attention, audio_input)
            pooled = tf.reshape(tf.concat((rgb_out,audio_out),axis=2),[-1,shape[2]])

        pooled = tf.nn.l2_normalize(pooled, dim=1)
        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=pooled,
            vocab_size=vocab_size,
            **unused_params)

class LstmExtendCombineModel(models.BaseModel):

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

        lstm_input = model_input
        lstm_shape = lstm_input.get_shape().as_list()

        ## Batch normalize the input
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)

        with tf.variable_scope("RNN"):
            outputs, state = tf.nn.dynamic_rnn(stacked_lstm, lstm_input,
                                               sequence_length=num_frames,
                                               swap_memory=True,
                                               dtype=tf.float32)
        W1 = tf.Variable(tf.truncated_normal([lstm_size,num_extend], stddev=0.1), name="W1")
        b1 = tf.Variable(tf.constant(0.1, shape=[num_extend]), name="b1")
        W2 = tf.Variable(tf.truncated_normal([lstm_shape[2],num_extend], stddev=0.1), name="W2")
        b2 = tf.Variable(tf.constant(0.1, shape=[num_extend]), name="b2")
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(W1))
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(b1))
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(W2))
        tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(b2))
        rnn_output = tf.reshape(outputs,[-1,lstm_size])
        rnn_input = tf.reshape(lstm_input,[-1,lstm_shape[2]])
        rnn_attention = tf.nn.softmax(tf.reshape(tf.nn.xw_plus_b(rnn_output,W1,b1)+tf.nn.xw_plus_b(rnn_input,W2,b2),[-1,lstm_shape[1],num_extend]),dim=1)*frames_bool
        rnn_attention = rnn_attention/tf.reduce_sum(rnn_attention, axis=1, keep_dims=True)
        rnn_attention = tf.transpose(rnn_attention,perm=[0,2,1])
        rnn_out = tf.einsum('aij,ajk->aik', rnn_attention, outputs)

        rnn_pooled = tf.reshape(rnn_out,[-1,lstm_size])
        aggregated_model = getattr(video_level_models, "MoeMix4Model")
        result = aggregated_model().create_model(
            model_input=rnn_pooled,
            vocab_size=vocab_size,
            **unused_params)

        final_probabilities = result["predictions"]
        probabilities_by_class = result["predictions_class"]
        final_probabilities = tf.reshape(final_probabilities,[-1,num_extend,vocab_size])
        probabilities_by_class = tf.reshape(probabilities_by_class,[-1,num_extend,vocab_size*FLAGS.moe_layers])
        result["predictions"] = tf.reduce_max(final_probabilities,axis=1)
        result["predictions_class"] = tf.reduce_max(probabilities_by_class,axis=1)
        return result

class LstmMoeModel(models.BaseModel):

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
        shape = model_input.get_shape().as_list()
        lstm_size = FLAGS.lstm_cells
        number_of_layers = FLAGS.lstm_layers
        num_mixtures = 2

        ## Batch normalize the input
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)

        with tf.variable_scope("RNN"):
            states = []
            layer_input = model_input
            for i in range(number_of_layers):
                outputs, state = tf.nn.dynamic_rnn(lstm_cell, layer_input,
                                               sequence_length=num_frames,
                                               swap_memory=True,
                                               dtype=tf.float32,scope="lstm-%s" % i)
                states.append(state)
                if i<number_of_layers-1:
                    outputs = tf.reshape(outputs,[-1,lstm_size])
                    gate_activations = slim.fully_connected(
                        outputs,
                        lstm_size * (num_mixtures + 1),
                        activation_fn=None,
                        biases_initializer=None,
                        weights_regularizer=slim.l2_regularizer(l2_penalty),
                        scope="gates-%s" % i)
                    expert_activations = slim.fully_connected(
                        outputs,
                        lstm_size * num_mixtures,
                        activation_fn=None,
                        weights_regularizer=slim.l2_regularizer(l2_penalty),
                        scope="experts-%s" % i)
                    gating_distribution = tf.nn.softmax(tf.reshape(
                        gate_activations,
                        [-1, num_mixtures + 1]))
                    expert_distribution = tf.reshape(
                        expert_activations,
                        [-1, num_mixtures])
                    final_probabilities = tf.reduce_sum(
                        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
                    layer_input = tf.reshape(final_probabilities,[-1,shape[1],lstm_size])

        state = tf.concat(states,axis=1)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=state,
            vocab_size=vocab_size,
            **unused_params)

class InputExtendModel(models.BaseModel):

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
        frames_range = frames_true*tf.reshape(tf.range(shape[1],dtype=tf.float32),[1,-1])
        frames_bool_all = tf.reshape(tf.where(tf.less_equal(frames_sum, frames_false), frames_true, frames_false),[-1,shape[1],1])
        frames_bools = []
        for i in range(num_extend):
            frame_start = tf.reshape((num_frames//num_extend)*i,[-1,1])
            frame_end = frame_start + tf.reshape((num_frames//num_extend)*2,[-1,1])
            bool_start = tf.ones(tf.shape(frames_sum))*tf.cast(frame_start,dtype=tf.float32)
            bool_end = tf.ones(tf.shape(frames_sum))*tf.cast(frame_end,dtype=tf.float32)
            frames_bool_start = tf.where(tf.greater_equal(frames_range, bool_start), frames_true, frames_false)
            frames_bool_end = tf.where(tf.less_equal(frames_range, bool_end), frames_true, frames_false)
            frames_bool = frames_bool_start*frames_bool_end
            frames_bools.append(frames_bool)
        frames_bools = tf.stack(frames_bools,axis=2)
        ## Batch normalize the input
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)

        with tf.variable_scope("RNN_Attention"):
            outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                               sequence_length=num_frames,
                                               swap_memory=True,
                                               dtype=tf.float32)
        with tf.variable_scope("RNN"):
            outputs_features, _ = tf.nn.dynamic_rnn(stacked_lstm, model_input,
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
        rnn_attention = tf.nn.softmax(tf.reshape(tf.nn.xw_plus_b(rnn_output,W1,b1)+tf.nn.xw_plus_b(rnn_input,W2,b2),[-1,shape[1],num_extend]),dim=1)*frames_bools
        rnn_attention = rnn_attention/tf.reduce_sum(rnn_attention, axis=1, keep_dims=True)
        rnn_attention = tf.transpose(rnn_attention,perm=[0,2,1])
        rnn_out = tf.einsum('aij,ajk->aik', rnn_attention, outputs_features)

        pooled = tf.reshape(rnn_out,[-1,lstm_size])

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=pooled,
            vocab_size=vocab_size,
            **unused_params)


class VideoFrameEvalModel(models.BaseModel):

    def create_model(self, model_input, vocab_size, num_frames, **unused_params):

        num_extend = FLAGS.moe_num_extend
        shape = model_input.get_shape().as_list()
        frames_sum = tf.reduce_sum(tf.abs(model_input),axis=2)
        random_inputs = []
        random_num_frames = []
        for i in range(num_extend):
            frames_index = tf.reshape(tf.range(shape[1],dtype=tf.float32),[1,shape[1]])
            frames_index = tf.ones(tf.shape(frames_sum))*frames_index
            frames_index = tf.transpose(tf.random_shuffle(tf.transpose(frames_index)))
            frames_index = frames_index[:,0:shape[1]//2]
            frames_index = tf.negative(frames_index)
            frames_index, indexes = tf.nn.top_k(frames_index, sorted=True, k=shape[1]//2)
            frames_index = tf.negative(frames_index)
            frames_valid = tf.cast(tf.reshape(num_frames,[-1,1]),dtype=tf.float32)
            frames_true = tf.ones(tf.shape(frames_index))
            frames_false = tf.zeros(tf.shape(frames_index))
            frames_bool = tf.where(tf.less(frames_index, frames_valid), frames_true, frames_false)
            random_num_frame = tf.reduce_sum(frames_bool,axis=1)
            random_num_frames.append(random_num_frame)

            batch_size = tf.shape(model_input)[0]
            batch_index = tf.tile(
                tf.expand_dims(tf.range(batch_size), 1), [1, shape[1]//2])
            index = tf.stack([batch_index, tf.cast(frames_index,dtype=tf.int32)], 2)
            random_input = tf.gather_nd(model_input, index)
            random_inputs.append(random_input)

        model_input = tf.stack(random_inputs,axis=1)
        random_num_frames = tf.stack(random_num_frames,axis=1)
        model_input = tf.reshape(model_input,[-1,shape[1]//2,shape[2]])
        random_num_frames = tf.reshape(random_num_frames,[-1])

        frames_sum = tf.reduce_sum(tf.abs(model_input),axis=2)
        frames_true = tf.ones(tf.shape(frames_sum))
        frames_false = tf.zeros(tf.shape(frames_sum))
        frames_bool = tf.reshape(tf.where(tf.greater(frames_sum, frames_false), frames_true, frames_false),[-1,shape[1]//2,1])

        model_input = tf.reduce_sum(model_input*frames_bool,axis=1)/(tf.reduce_sum(frames_bool,axis=1)+1e-6)
        model_input = tf.nn.l2_normalize(model_input,dim=1)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        result_frames = aggregated_model().create_model(
            model_input=model_input,
            vocab_size=vocab_size,
            **unused_params)

        predictions_frames = result_frames["predictions"]
        predictions_frames = tf.reshape(predictions_frames,[-1,num_extend,vocab_size])
        final_probabilities = tf.reduce_mean(predictions_frames,axis=1)

        return {"predictions": final_probabilities}

class FrameAttentionEvalModel(models.BaseModel):

    def create_model(self, model_input, vocab_size, num_frames, **unused_params):

        num_extend = FLAGS.moe_num_extend
        shape = model_input.get_shape().as_list()
        frames_sum = tf.reduce_sum(tf.abs(model_input),axis=2)
        random_inputs = []
        random_num_frames = []
        for i in range(num_extend):
            frames_index = tf.reshape(tf.range(shape[1],dtype=tf.float32),[1,shape[1]])
            frames_index = tf.ones(tf.shape(frames_sum))*frames_index
            frames_index = tf.transpose(tf.random_shuffle(tf.transpose(frames_index)))
            frames_index = frames_index[:,0:shape[1]//2]
            frames_index = tf.negative(frames_index)
            frames_index, indexes = tf.nn.top_k(frames_index, sorted=True, k=shape[1]//2)
            frames_index = tf.negative(frames_index)
            frames_valid = tf.cast(tf.reshape(num_frames,[-1,1]),dtype=tf.float32)
            frames_true = tf.ones(tf.shape(frames_index))
            frames_false = tf.zeros(tf.shape(frames_index))
            frames_bool = tf.where(tf.less(frames_index, frames_valid), frames_true, frames_false)
            random_num_frame = tf.reduce_sum(frames_bool,axis=1)
            random_num_frames.append(random_num_frame)

            batch_size = tf.shape(model_input)[0]
            batch_index = tf.tile(
                tf.expand_dims(tf.range(batch_size), 1), [1, shape[1]//2])
            index = tf.stack([batch_index, tf.cast(frames_index,dtype=tf.int32)], 2)
            random_input = tf.gather_nd(model_input, index)
            random_inputs.append(random_input)

        model_input = tf.stack(random_inputs,axis=1)
        random_num_frames = tf.stack(random_num_frames,axis=1)
        model_input = tf.reshape(model_input,[-1,shape[1]//2,shape[2]])
        random_num_frames = tf.reshape(random_num_frames,[-1])

        result_frames = LstmExtendModel().create_model(
            model_input=model_input,
            vocab_size=vocab_size,
            num_frames=random_num_frames,
            **unused_params)

        predictions_frames = result_frames["predictions"]
        predictions_frames = tf.reshape(predictions_frames,[-1,num_extend,vocab_size])
        predictions_frames = tf.clip_by_value(predictions_frames,0.0,1.0)
        final_probabilities = tf.reduce_mean(predictions_frames,axis=1)

        return {"predictions": final_probabilities}

class FrameRandomEvalModel(models.BaseModel):

    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        lstm_size = FLAGS.lstm_cells
        number_of_layers = FLAGS.lstm_layers
        num_extend = FLAGS.moe_num_extend
        shape = model_input.get_shape().as_list()
        frames_sum = tf.reduce_sum(tf.abs(model_input),axis=2)
        random_inputs = []
        random_num_frames = []
        for i in range(num_extend):
            frames_index = tf.reshape(tf.range(shape[1],dtype=tf.float32),[1,shape[1]])
            frames_index = tf.ones(tf.shape(frames_sum))*frames_index
            frames_index = tf.transpose(tf.random_shuffle(tf.transpose(frames_index)))
            frames_index = frames_index[:,0:shape[1]//2]
            frames_index = tf.negative(frames_index)
            frames_index, indexes = tf.nn.top_k(frames_index, sorted=True, k=shape[1]//2)
            frames_index = tf.negative(frames_index)
            frames_valid = tf.cast(tf.reshape(num_frames,[-1,1]),dtype=tf.float32)
            frames_true = tf.ones(tf.shape(frames_index))
            frames_false = tf.zeros(tf.shape(frames_index))
            frames_bool = tf.where(tf.less(frames_index, frames_valid), frames_true, frames_false)
            random_num_frame = tf.reduce_sum(frames_bool,axis=1)
            random_num_frames.append(random_num_frame)

            batch_size = tf.shape(model_input)[0]
            batch_index = tf.tile(
                tf.expand_dims(tf.range(batch_size), 1), [1, shape[1]//2])
            index = tf.stack([batch_index, tf.cast(frames_index,dtype=tf.int32)], 2)
            random_input = tf.gather_nd(model_input, index)
            random_inputs.append(random_input)

        model_input = tf.stack(random_inputs,axis=1)
        random_num_frames = tf.stack(random_num_frames,axis=1)
        model_input = tf.reshape(model_input,[-1,shape[1]//2,shape[2]])
        random_num_frames = tf.reshape(random_num_frames,[-1])
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
                                               sequence_length=random_num_frames,
                                               swap_memory=True,
                                               dtype=tf.float32)
        state_c = tf.concat(map(lambda x: x.c, state), axis=1)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        result_frames = aggregated_model().create_model(
            model_input=state_c,
            vocab_size=vocab_size,
            **unused_params)

        #final_probabilities = result_frames["predictions"]

        predictions_frames = result_frames["predictions"]
        predictions_frames = tf.reshape(predictions_frames,[-1,num_extend,vocab_size])
        final_probabilities = tf.reduce_mean(predictions_frames,axis=1)

        return {"predictions": final_probabilities}

class CnnDCCDistillChainModel(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def cnn(self,
            model_input,
            l2_penalty=1e-8,
            num_filters = [1024, 1024, 1024],
            filter_sizes = [1,2,3],
            sub_scope="",
            **unused_params):
        max_frames = model_input.get_shape().as_list()[1]
        num_features = model_input.get_shape().as_list()[2]

        shift_inputs = []
        for i in range(max(filter_sizes)):
            if i == 0:
                shift_inputs.append(model_input)
            else:
                shift_inputs.append(tf.pad(model_input, paddings=[[0,0],[i,0],[0,0]])[:,:max_frames,:])

        cnn_outputs = []
        for nf, fs in zip(num_filters, filter_sizes):
            sub_input = tf.concat(shift_inputs[:fs], axis=2)
            sub_filter = tf.get_variable(sub_scope+"cnn-filter-len%d"%fs,
                                         shape=[num_features*fs, nf], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                         regularizer=tf.contrib.layers.l2_regularizer(l2_penalty))
            cnn_outputs.append(tf.einsum("ijk,kl->ijl", sub_input, sub_filter))

        cnn_output = tf.concat(cnn_outputs, axis=2)
        return cnn_output

    def create_model(self, model_input, vocab_size, num_frames, num_mixtures=None,
                     l2_penalty=1e-8, sub_scope="", distill_labels=None, original_input=None, **unused_params):
        num_layers = FLAGS.moe_layers
        relu_cells = 256
        max_frames = model_input.get_shape().as_list()[1]
        relu_layers = []
        support_predictions = []

        mask = self.get_mask(max_frames, num_frames)
        mean_input = tf.einsum("ijk,ij->ik", model_input, mask) \
                     / tf.expand_dims(tf.cast(num_frames, dtype=tf.float32), dim=1)
        mean_relu = slim.fully_connected(
            mean_input,
            relu_cells,
            activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope=sub_scope+"mean-relu")
        mean_relu_norm = tf.nn.l2_normalize(mean_relu, dim=1)
        relu_layers.append(mean_relu_norm)

        cnn_output = self.cnn(model_input, num_filters=[relu_cells,relu_cells,relu_cells*2], filter_sizes=[1,2,3], sub_scope=sub_scope+"cnn0")
        max_cnn_output = tf.reduce_max(cnn_output, axis=1)
        normalized_cnn_output = tf.nn.l2_normalize(max_cnn_output, dim=1)
        next_input = normalized_cnn_output

        for layer in range(num_layers):
            if layer==0:
                sub_prediction = self.sub_model(next_input, vocab_size, distill_labels=distill_labels, sub_scope=sub_scope+"prediction-%d"%layer)
            else:
                sub_prediction = self.sub_model(next_input, vocab_size, sub_scope=sub_scope+"prediction-%d"%layer)
            support_predictions.append(sub_prediction)

            sub_relu = slim.fully_connected(
                sub_prediction,
                relu_cells,
                activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope=sub_scope+"relu-%d"%layer)
            relu_norm = tf.nn.l2_normalize(sub_relu, dim=1)
            relu_layers.append(relu_norm)

            cnn_output = self.cnn(model_input, num_filters=[relu_cells,relu_cells,relu_cells*2], filter_sizes=[1,2,3], sub_scope=sub_scope+"cnn%d"%(layer+1))
            max_cnn_output = tf.reduce_max(cnn_output, axis=1)
            normalized_cnn_output = tf.nn.l2_normalize(max_cnn_output, dim=1)
            next_input = tf.concat([normalized_cnn_output] + relu_layers, axis=1)

        main_predictions = self.sub_model(next_input, vocab_size, sub_scope=sub_scope+"-main")
        support_predictions = tf.concat(support_predictions, axis=1)
        return {"predictions": main_predictions, "predictions_class": support_predictions}

    def sub_model(self, model_input, vocab_size, num_mixtures=None,
                  l2_penalty=1e-8, sub_scope="", distill_labels=None,**unused_params):
        num_mixtures = num_mixtures or FLAGS.moe_num_mixtures
        class_size = 256
        if distill_labels is not None:
            class_input = slim.fully_connected(
                distill_labels,
                class_size,
                activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="class_inputs")
            class_input = tf.nn.l2_normalize(class_input, dim=1)
            model_input = tf.concat((model_input, class_input),axis=1)
        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates-"+sub_scope)
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts-"+sub_scope)

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        final_probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                         [-1, vocab_size])
        return final_probabilities

    def get_mask(self, max_frames, num_frames):
        mask_array = []
        for i in range(max_frames + 1):
            tmp = [0.0] * max_frames
            for j in range(i):
                tmp[j] = 1.0
            mask_array.append(tmp)
        mask_array = np.array(mask_array)
        mask_init = tf.constant_initializer(mask_array)
        mask_emb = tf.get_variable("mask_emb", shape = [max_frames + 1, max_frames],
                                   dtype = tf.float32, trainable = False, initializer = mask_init)
        mask = tf.nn.embedding_lookup(mask_emb, num_frames)
        return mask