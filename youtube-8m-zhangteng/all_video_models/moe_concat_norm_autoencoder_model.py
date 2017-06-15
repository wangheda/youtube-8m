import math

import models
import tensorflow as tf
import numpy as np
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS


class MoeConcatNormAutoencoderModel(models.BaseModel):
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
        class_expert_distribution = tf.reshape(class_expert_activations,[-1,num_mixtures])

        probabilities_by_class = tf.reduce_sum(
            class_gating_distribution[:, :num_mixtures] * class_expert_distribution, 1)

        probabilities_by_class = tf.reshape(probabilities_by_class,
                                            [-1, class_size])

        hidden_mean = tf.reduce_mean(probabilities_by_class,axis=1,keep_dims=True)
        hidden_std = tf.sqrt(tf.reduce_mean(tf.square(probabilities_by_class-hidden_mean),axis=1,keep_dims=True))
        probabilities_by_class = (probabilities_by_class-hidden_mean)/(hidden_std+1e-6)
        hidden_2 = tf.nn.relu(probabilities_by_class)

        vars = np.loadtxt(FLAGS.autoencoder_dir+'autoencoder_layer%d.model' % FLAGS.encoder_layers)
        weights = tf.constant(vars[:-1,:],dtype=tf.float32)
        bias = tf.reshape(tf.constant(vars[-1,:],dtype=tf.float32),[-1])
        class_output = tf.nn.relu(tf.nn.xw_plus_b(hidden_2,weights,bias))

        #class_output = probabilities_by_class

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


        return {"predictions": final_probabilities, "predictions_encoder": probabilities_by_class}