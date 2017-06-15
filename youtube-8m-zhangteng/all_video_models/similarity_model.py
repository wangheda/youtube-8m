import math

import models
import tensorflow as tf
import numpy as np
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim
FLAGS = flags.FLAGS


class SimilarityModel(models.BaseModel):
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
        embedding_size = model_input.get_shape().as_list()[1]

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)

        model_input = tf.maximum(model_input,tf.zeros_like(model_input))
        expert_distribution = []
        for i in range(num_mixtures):
            embeddings = tf.Variable(tf.truncated_normal([vocab_size, embedding_size],stddev=0.1))
            tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(embeddings))
            embeddings = tf.maximum(embeddings,tf.zeros_like(embeddings))
            norm_embeddings = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = tf.div(embeddings, norm_embeddings)
            norm_input = tf.sqrt(tf.reduce_sum(tf.square(model_input), 1, keep_dims=True))
            normalized_input = tf.div(model_input,norm_input)
            similarity = tf.matmul(normalized_input, normalized_embeddings, transpose_b=True)*2
            expert_distribution.append(similarity)

        expert_distribution = tf.stack(expert_distribution,axis=2)
        expert_distribution = tf.reshape(expert_distribution,[-1,num_mixtures])

        probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)

        probabilities_by_class_and_batch = tf.reshape(probabilities_by_class_and_batch,
                                                      [-1, vocab_size])

        final_probabilities = tf.reshape(probabilities_by_class_and_batch,
                                         [-1, vocab_size])

        return {"predictions": final_probabilities}