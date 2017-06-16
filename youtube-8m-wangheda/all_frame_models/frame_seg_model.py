import sys
import models
import model_utils
import math
import numpy as np
import video_level_models
import tensorflow as tf
import utils
import tensorflow.contrib.slim as slim
from tensorflow import flags
FLAGS = flags.FLAGS

class FrameSegModel(models.BaseModel):

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   l2_penalty=1e-8,
                   **unused_params):

    relu_cells = FLAGS.frame_seg_relu_cells
    float_num_frames = tf.cast(num_frames, tf.float32)
    divisions = [0.0, 0.1, 0.5, 0.9, 1.0]

    relu_layers = []
    for i in xrange(4):
      frame_start = float_num_frames * divisions[i]
      frame_end = float_num_frames * divisions[i+1]
      mean_frame = self.frame_mean(model_input, frame_start, frame_end)
      mean_frame = tf.nn.l2_normalize(mean_frame, dim=1)
      mean_relu = slim.fully_connected(
            mean_frame,
            relu_cells,
            activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="relu%d"%i)
      relu_layers.append(tf.nn.l2_normalize(mean_relu, dim=1))

    relu_layer = tf.concat(relu_layers, axis=1)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=relu_layer,
        original_input=model_input,
        vocab_size=vocab_size,
        **unused_params)

  def frame_mean(self, model_input, frame_start, 
                 frame_end, **unused_params):
    max_frames = model_input.shape.as_list()[-2]

    frame_start = tf.cast(frame_start, tf.int32)
    frame_end = tf.cast(frame_end, tf.int32)

    frame_length = tf.expand_dims(tf.cast(frame_end - frame_start, tf.float32), axis=1)
    frame_mask = tf.sequence_mask(frame_end, maxlen=max_frames, dtype=tf.float32) \
                 - tf.sequence_mask(frame_start, maxlen=max_frames, dtype=tf.float32)
    mean_frame = tf.einsum("ijk,ij->ik", model_input, frame_mask) / (0.1 + frame_length)
    return mean_frame

