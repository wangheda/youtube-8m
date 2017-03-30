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

import tensorflow.contrib.slim as slim
from tensorflow import flags

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
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")

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
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ],
            state_is_tuple=False)

    with tf.variable_scope("RNN"):
      outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                         sequence_length=num_frames,
                                         swap_memory=True,
                                         dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class LstmFrameModel(models.BaseModel):

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
        frames_bool_norm = frames_bool/tf.reduce_sum(frames_bool,axis=1,keep_dims=True)

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

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)

        video_input = tf.reduce_sum(model_input*frames_bool_norm,axis=1)
        result = aggregated_model().create_model(
            model_input=video_input,
            vocab_size=vocab_size,
            **unused_params)
        result["bottleneck"] = tf.reduce_sum(frames_bool_norm*outputs,axis=1)
        sigmoid_input = tf.reduce_mean(tf.reshape(outputs,[-1,FLAGS.stride_size,lstm_size]),axis=1)
        frames_bool = frames_bool[:,0:shape[1]:FLAGS.stride_size,:]
        output = slim.fully_connected(
            sigmoid_input, vocab_size, activation_fn=tf.nn.sigmoid,
            weights_regularizer=slim.l2_regularizer(1e-8))
        perdiction_frames = output*tf.reshape(frames_bool,[-1,1])
        result["prediction_frames"] = perdiction_frames
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
            #pooled = tf.reshape(cnn_output,[-1,num_extend*num_filters])

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
        return aggregated_model().create_model(
            model_input=pooled,
            vocab_size=vocab_size,
            **unused_params)

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