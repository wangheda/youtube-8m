import math

import models
import tensorflow as tf
import numpy as np
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS


class MoeMaxConcatModel(models.BaseModel):
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
        class_size = 25

        class_input = slim.fully_connected(
            model_input,
            model_input.get_shape().as_list()[1],
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
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        probabilities_by_class = tf.reduce_sum(
            class_gating_distribution[:, :num_mixtures] * class_expert_distribution, 1)
        probabilities_by_class = tf.reshape(probabilities_by_class,
                                            [-1, class_size])

        vocab_input = tf.concat((model_input,probabilities_by_class), axis=1)
        gate_activations = slim.fully_connected(
            vocab_input,
            vocab_size * (num_mixtures+1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
        expert_activations = slim.fully_connected(
            vocab_input,
            vocab_size*num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")
        expert_others = slim.fully_connected(
            vocab_input,
            vocab_size,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="others")

        expert_activations = tf.reshape(expert_activations,[-1,vocab_size,num_mixtures])
        forward_indices = []
        backward_indices = []
        for i in range(num_mixtures):
            forward_indice = np.arange(vocab_size)
            np.random.seed(i)
            np.random.shuffle(forward_indice)
            backward_indice = np.argsort(forward_indice,axis=None)
            forward_indices.append(forward_indice)
            backward_indices.append(backward_indice)

        forward_indices = tf.constant(np.stack(forward_indices,axis=1),dtype=tf.int32)*num_mixtures + tf.reshape(tf.range(num_mixtures),[1,-1])
        backward_indices = tf.constant(np.stack(backward_indices,axis=1),dtype=tf.int32)*num_mixtures + tf.reshape(tf.range(num_mixtures),[1,-1])
        forward_indices = tf.stop_gradient(tf.reshape(forward_indices,[-1]))
        backward_indices = tf.stop_gradient(tf.reshape(backward_indices,[-1]))

        expert_activations = tf.transpose(tf.reshape(expert_activations,[-1,vocab_size*num_mixtures]))
        expert_activations = tf.transpose(tf.gather(expert_activations,forward_indices))
        expert_activations = tf.reshape(expert_activations,[-1,vocab_size,num_mixtures])

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures+1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_softmax = tf.transpose(expert_activations,perm=[0,2,1])
        expert_softmax = tf.concat((tf.reshape(expert_softmax,[-1,num_mixtures]),tf.reshape(expert_others,[-1,1])),axis=1)
        expert_distribution = tf.nn.softmax(tf.reshape(
            expert_softmax,
            [-1, num_mixtures+1]))  # (Batch * #Labels) x num_mixtures

        expert_distribution = tf.reshape(expert_distribution[:,:num_mixtures],[-1,num_mixtures,vocab_size])
        expert_distribution = tf.reshape(tf.transpose(expert_distribution,perm=[0,2,1]),[-1,vocab_size*num_mixtures])

        expert_distribution = tf.transpose(tf.gather(tf.transpose(expert_distribution),backward_indices))
        expert_distribution = tf.reshape(expert_distribution,[-1,num_mixtures])
        probabilities_by_class_and_batch = tf.reduce_sum(gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        final_probabilities = tf.reshape(probabilities_by_class_and_batch,[-1, vocab_size])

        return {"predictions": final_probabilities, "predictions_class": probabilities_by_class}