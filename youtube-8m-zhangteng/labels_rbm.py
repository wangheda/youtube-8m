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

"""Contains model definitions."""
import math

import models
import tensorflow as tf

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "hidden_size", 25,
    "The szie of hidden layers.")

class RbmModel(models.BaseModel):
    """Logistic model with L2 regularization."""

    def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
        """Creates a logistic model.

        Args:
          model_input: 'batch' x 'num_features' matrix of input features.
          vocab_size: The number of classes in the dataset.

        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes."""

        input_size = vocab_size
        output_size = FLAGS.hidden_size
        with tf.name_scope("rbm"):
            self.weights = tf.Variable(tf.truncated_normal([input_size, output_size],
                                    stddev=1.0 / math.sqrt(float(input_size))), name="weights")
            self.v_bias = tf.Variable(tf.zeros([input_size]), name="v_bias")
            self.h_bias = tf.Variable(tf.zeros([output_size]), name="h_bias")
            tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(self.weights))
            tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(self.v_bias))
            tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=l2_penalty*tf.nn.l2_loss(self.h_bias))

    def propup(self, visible):
        return tf.nn.sigmoid(tf.matmul(visible, self.weights) + self.h_bias)

    def propdown(self, hidden):
        return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(self.weights)) + self.v_bias)

    def cd1(self, visibles, learning_rate=0.1):
        h_start = self.propup(visibles)
        v_end = self.propdown(h_start)
        h_end = self.propup(v_end)
        w_positive_grad = tf.matmul(tf.transpose(visibles), h_start)
        w_negative_grad = tf.matmul(tf.transpose(v_end), h_end)
        update_w = self.weights.assign_add(learning_rate * (w_positive_grad - w_negative_grad))
        update_vb = self.v_bias.assign_add(learning_rate * tf.reduce_mean(visibles - v_end, 0))
        update_hb = self.h_bias.assign_add(learning_rate * tf.reduce_mean(h_start - h_end, 0))
        return [update_w, update_vb, update_hb]

    def reconstruction_error(self, model_input):
        err = tf.stop_gradient(model_input - self.gibbs_vhv(model_input)[1])
        return tf.reduce_sum(err * err)

    def sample_prob(self,probs):
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(probs.get_shape())))
    def sample_h_given_v(self, v_sample):
        return self.sample_prob(self.propup(v_sample))
    def sample_v_given_h(self, h_sample):
        return self.sample_prob(self.propdown(h_sample))
    def gibbs_hvh(self, h0_sample):
        v_sample = self.sample_v_given_h(h0_sample)
        h_sample = self.sample_h_given_v(v_sample)
        return [v_sample, h_sample]
    def gibbs_vhv(self, v0_sample):
        h_sample = self.sample_h_given_v(v0_sample)
        v_sample = self.sample_v_given_h(h_sample)
        return  [h_sample, v_sample]