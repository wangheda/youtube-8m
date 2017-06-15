import math

import models
import tensorflow as tf
import numpy as np
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS


class MoeConcatAutoencoderModel(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
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
        class_size = FLAGS.encoder_size
        hidden_channels = FLAGS.hidden_channels
        shape = model_input.get_shape().as_list()[1]

        class_input = slim.fully_connected(
            model_input,
            shape,
            activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="class_inputs")

        class_gate_activations = slim.fully_connected(
            class_input,
            class_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="class_gates")
        class_expert_activations = slim.fully_connected(
            class_input,
            class_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="class_experts")

        class_gating_distribution = tf.nn.softmax(tf.reshape(
            class_gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        class_expert_distribution = tf.nn.sigmoid(tf.reshape(
            class_expert_activations,
            [-1,class_size, num_mixtures]))  # (Batch * #Labels) x num_mixtures
        class_expert_distribution = tf.reshape(class_expert_distribution,[-1,num_mixtures])

        probabilities_by_class = tf.reduce_sum(
            class_gating_distribution[:, :num_mixtures] * class_expert_distribution, 1)
        """
        class_expert_activations = slim.fully_connected(
            class_input,
            class_size,
            activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="class_experts")
        probabilities_by_class = slim.fully_connected(
            class_expert_activations,
            class_size,
            activation_fn=tf.nn.softmax,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="probabilities_by_class")"""

        probabilities_by_class = tf.reshape(probabilities_by_class,
                                            [-1, class_size])

        vars = np.loadtxt(FLAGS.autoencoder_dir+'autoencoder_layer%d.model' % FLAGS.encoder_layers)
        weights = tf.constant(vars[:-1,:],dtype=tf.float32)
        bias = tf.reshape(tf.constant(vars[-1,:],dtype=tf.float32),[-1])
        class_output = tf.nn.relu(tf.nn.xw_plus_b(probabilities_by_class,weights,bias))

        class_output = tf.nn.l2_normalize(class_output,dim=1)*tf.sqrt(tf.cast(class_size,dtype=tf.float32)/shape)

        vocab_input = tf.concat((model_input, class_output), axis=1)
        gate_activations = slim.fully_connected(
            vocab_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
        expert_activations = slim.fully_connected(
            vocab_input,
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

        probabilities_by_vocab = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        probabilities_by_vocab = tf.reshape(probabilities_by_vocab,
                                            [-1, vocab_size])

        final_probabilities = probabilities_by_vocab

        """
        final_probabilities = tf.reshape(probabilities_by_class,[-1,class_size*hidden_channels])
        for i in range(FLAGS.encoder_layers, FLAGS.encoder_layers*2):
            var_i = np.loadtxt(FLAGS.autoencoder_dir+'autoencoder_layer%d.model' % i)
            weight_i = tf.constant(var_i[:-1,:],dtype=tf.float32)
            bias_i = tf.reshape(tf.constant(var_i[-1,:],dtype=tf.float32),[-1])
            final_probabilities = tf.nn.xw_plus_b(final_probabilities,weight_i,bias_i)
            if i<FLAGS.encoder_layers*2-1:
                final_probabilities = tf.nn.relu(final_probabilities)
            else:
                final_probabilities = tf.nn.sigmoid(final_probabilities)"""

        return {"predictions": final_probabilities, "predictions_encoder": probabilities_by_class}