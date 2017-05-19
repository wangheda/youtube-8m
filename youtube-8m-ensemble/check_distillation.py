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
"""Binary for evaluating Tensorflow models on the YouTube-8M dataset."""

import time

import numpy as np
import eval_util
import losses
import ensemble_level_models
import readers
import tensorflow as tf
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
import utils

FLAGS = flags.FLAGS

if __name__ == "__main__":
  # Dataset flags.
  flags.DEFINE_string(
      "eval_data_patterns", "",
      "File globs defining the evaluation dataset in tensorflow.SequenceExample format.")
  flags.DEFINE_string(
      "input_data_pattern", None,
      "File globs for original model input.")
  flags.DEFINE_string("feature_names", "predictions", "Name of the feature "
                      "to use for training.")
  flags.DEFINE_string("feature_sizes", "4716", "Length of the feature vectors.")

  # Model flags.
  flags.DEFINE_integer("batch_size", 1024,
                       "How many examples to process per batch.")

def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)


def get_input_evaluation_tensors(reader,
                                 data_pattern,
                                 batch_size=256):
  logging.info("Using batch size of " + str(batch_size) + " for evaluation.")
  with tf.name_scope("eval_input"):
    files = gfile.Glob(data_pattern)
    if not files:
      print data_pattern, files
      raise IOError("Unable to find the evaluation files.")
    logging.info("number of evaluation files: " + str(len(files)))
    files.sort()
    filename_queue = tf.train.string_input_producer(
        files, shuffle=False, num_epochs=1)
    eval_data = reader.prepare_reader(filename_queue)
    return tf.train.batch(
        eval_data,
        batch_size=batch_size,
        capacity=3 * batch_size,
        allow_smaller_final_batch=True,
        enqueue_many=True)


def build_graph(all_readers,
                input_reader,
                input_data_pattern,
                all_eval_data_patterns,
                batch_size=256):

  original_video_id, original_input, unused_labels_batch, unused_num_frames = (
      get_input_evaluation_tensors(
          input_reader,
          input_data_pattern,
          batch_size=batch_size))
  
  video_id_equal_tensors = []
  model_input_tensor = None
  input_distance_tensors = []

  model_label_tensor = tf.cast(unused_labels_batch, dtype=tf.float32)
  label_distance_tensors = []

  for reader, data_pattern in zip(all_readers, all_eval_data_patterns):
    video_id, model_input_raw, labels_batch, unused_num_frames = (
        get_input_evaluation_tensors(
            reader,
            data_pattern,
            batch_size=batch_size))

    video_id_equal_tensors.append(tf.reduce_sum(tf.cast(tf.not_equal(original_video_id, video_id), dtype=tf.float32)))
    input_distance_tensors.append(tf.reduce_mean(tf.reduce_sum(tf.square(original_input - model_input_raw), axis=1)))

    labels_batch = tf.cast(labels_batch, dtype=tf.float32)
    x = model_input_raw
    y = labels_batch
    ce = - y * tf.log(x + 1e-7) - (1.0 - y) * tf.log(1.0 + 1e-7 - x)
    label_distance_tensors.append(tf.reduce_mean(tf.reduce_sum(ce, axis=1)))

  video_id_equal_tensor = tf.stack(video_id_equal_tensors)
  input_distance_tensor = tf.stack(input_distance_tensors)
  label_distance_tensor = tf.stack(label_distance_tensors)

  tf.add_to_collection("model_input", model_input_tensor)
  tf.add_to_collection("video_id_equal", video_id_equal_tensor)
  tf.add_to_collection("input_distance", input_distance_tensor)
  tf.add_to_collection("label_distance", label_distance_tensor)


def check_loop(model_input, video_id_equal, input_distance, label_distance, all_patterns):

  with tf.Session() as sess:
    sess.run([tf.local_variables_initializer()])

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(
            sess, coord=coord, daemon=True,
            start=True))

      examples_processed = 0
      while not coord.should_stop():
        batch_start_time = time.time()

        model_input_val, video_id_equal_val, input_distance_val, label_distance_val = sess.run([model_input, video_id_equal, input_distance, label_distance])
        print "model_input.max", np.max(model_input_val)
        print "model_input.min", np.min(model_input_val)
        print "input_distance_val", input_distance_val
        print "label_distance_val", label_distance_val
        for i in xrange(video_id_equal_val.shape[0]):
          if video_id_equal_val[i] > 0:
            print "%d discrepancies in %s" % (int(video_id_equal_val[i]), all_patterns[i])

        seconds_per_batch = time.time() - batch_start_time
        example_per_second = video_id_equal_val.shape[0] / seconds_per_batch
        examples_processed += video_id_equal_val.shape[0]

        logging.info("examples_processed: %d", examples_processed)

    except tf.errors.OutOfRangeError as e:
      logging.info(
          "Done with batched inference. Now calculating global performance "
          "metrics.")
    except Exception as e:  # pylint: disable=broad-except
      logging.info("Unexpected exception: " + str(e))
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

    return global_step_val


def check_video_id():
  tf.set_random_seed(0)  # for reproducibility
  with tf.Graph().as_default():
    # convert feature_names and feature_sizes to lists of values
    feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
        FLAGS.feature_names, FLAGS.feature_sizes)

    # prepare a reader for each single model prediction result
    all_readers = []

    all_patterns = FLAGS.eval_data_patterns
    all_patterns = map(lambda x: x.strip(), all_patterns.strip().strip(",").split(","))
    for i in xrange(len(all_patterns)):
      reader = readers.EnsembleReader(
          feature_names=feature_names, feature_sizes=feature_sizes)
      all_readers.append(reader)

    input_reader = None
    input_data_pattern = None
    if FLAGS.input_data_pattern is not None:
      input_reader = readers.EnsembleReader(
          feature_names=["mean_rgb","mean_audio"], feature_sizes=[1024,128])
      input_data_pattern = FLAGS.input_data_pattern

    if FLAGS.eval_data_patterns is "":
      raise IOError("'eval_data_patterns' was not specified. " +
                     "Nothing to evaluate.")

    build_graph(
        all_readers=all_readers,
        input_reader=input_reader,
        input_data_pattern=input_data_pattern,
        all_eval_data_patterns=all_patterns,
        batch_size=FLAGS.batch_size)

    logging.info("built evaluation graph")
    video_id_equal = tf.get_collection("video_id_equal")[0]
    model_input = tf.get_collection("model_input")[0]
    input_distance = tf.get_collection("input_distance")[0]
    label_distance = tf.get_collection("label_distance")[0]

    check_loop(model_input, video_id_equal, input_distance, label_distance, all_patterns)


def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)
  print("tensorflow version: %s" % tf.__version__)
  check_video_id()


if __name__ == "__main__":
  app.run()

