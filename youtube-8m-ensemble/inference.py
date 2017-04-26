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

"""Binary for generating predictions over a set of videos."""

import os
import time

import numpy
import tensorflow as tf

from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging

import eval_util
import losses
import readers
import utils

FLAGS = flags.FLAGS

if __name__ == '__main__':
  flags.DEFINE_string("model_checkpoint_path", None,
                      "The file path to load the model from.")
  flags.DEFINE_string("output_file", "",
                      "The file to save the predictions to.")
  flags.DEFINE_string(
      "input_data_patterns", "",
      "File globs defining the evaluation dataset in tensorflow.SequenceExample format.")
  flags.DEFINE_string("feature_names", "predictions", "Name of the feature "
                      "to use for training.")
  flags.DEFINE_string("feature_sizes", "4716", "Length of the feature vectors.")

  # Model flags.
  flags.DEFINE_integer(
      "batch_size", 1024,
      "How many examples to process per batch.")

  # Other flags.
  flags.DEFINE_integer("top_k", 20,
                       "How many predictions to output per video.")

def format_lines(video_ids, predictions, top_k):
  batch_size = len(video_ids)
  for video_index in range(batch_size):
    top_indices = numpy.argpartition(predictions[video_index], -top_k)[-top_k:]
    line = [(class_index, predictions[video_index][class_index])
            for class_index in top_indices]
    line = sorted(line, key=lambda p: -p[1])
    yield video_ids[video_index].decode('utf-8') + "," + " ".join("%i %f" % pair
                                                  for pair in line) + "\n"


def get_input_data_tensors(reader, 
                           data_pattern, 
                           batch_size, 
                           num_readers=1):
  with tf.name_scope("input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find input files. data_pattern='" +
                    data_pattern + "'")
    logging.info("number of input files: " + str(len(files)))
    files.sort()
    filename_queue = tf.train.string_input_producer(
        files, num_epochs=1, shuffle=False)
    input_data = reader.prepare_reader(filename_queue)
    video_id_batch, video_batch, unused_labels, unused_num_frames = (
        tf.train.batch(input_data,
                       batch_size=batch_size,
                       allow_smaller_final_batch = True,
                       enqueue_many=True))
    return video_id_batch, video_batch

def inference(all_readers, 
              model_checkpoint, 
              all_data_patterns, 
              out_file_location, 
              batch_size, 
              top_k):
  with tf.Session() as sess, gfile.Open(out_file_location, "w+") as out_file:
    # get data tensor: video_id_batch, video_batch = get_tensor
    video_batch = []
    video_id_batch = None
    for reader, data_pattern in zip(all_readers, all_data_patterns):
      video_id_tensor, video_tensor = (
          get_input_data_tensors(
              reader,
              data_pattern,
              batch_size=batch_size))
      if video_id_batch is None:
        video_id_batch = video_id_tensor
      video_batch.append(tf.expand_dims(video_tensor, axis=2))
    video_batch = tf.concat(video_batch, axis=2)

    if model_checkpoint is None:
      raise Exception("unable to find a checkpoint at location: %s" % train_dir)
    else:
      meta_graph_location = model_checkpoint + ".meta"
      logging.info("loading meta-graph: " + meta_graph_location)
    saver = tf.train.import_meta_graph(meta_graph_location, clear_devices=True)
    logging.info("restoring variables from " + latest_checkpoint)
    saver.restore(sess, latest_checkpoint)

    input_tensor = tf.get_collection("input_batch_raw")[0]
    predictions_tensor = tf.get_collection("predictions")[0]

    # Workaround for num_epochs issue.
    def set_up_init_ops(variables):
      init_op_list = []
      for variable in list(variables):
        if "train_input" in variable.name:
          init_op_list.append(tf.assign(variable, 1))
          variables.remove(variable)
      init_op_list.append(tf.variables_initializer(variables))
      return init_op_list

    sess.run(set_up_init_ops(tf.get_collection_ref(
        tf.GraphKeys.LOCAL_VARIABLES)))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    num_examples_processed = 0
    start_time = time.time()
    out_file.write("VideoId,LabelConfidencePairs\n")

    try:
      while not coord.should_stop():
          video_id_batch_val, video_batch_val = sess.run([video_id_batch, video_batch])
          predictions_val, = sess.run([predictions_tensor], feed_dict={input_tensor: video_batch_val})
          now = time.time()
          num_examples_processed += len(video_batch_val)
          num_classes = predictions_val.shape[1]
          logging.info("num examples processed: " + str(num_examples_processed) + " elapsed seconds: " + "{0:.2f}".format(now-start_time))
          for line in format_lines(video_id_batch_val, predictions_val, top_k):
            out_file.write(line)
          out_file.flush()

    except tf.errors.OutOfRangeError:
        logging.info('Done with inference. The output file was written to ' + out_file_location)
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)

  # convert feature_names and feature_sizes to lists of values
  feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
      FLAGS.feature_names, FLAGS.feature_sizes)

  # prepare a reader for each single model prediction result
  all_readers = []

  all_patterns = FLAGS.input_data_patterns
  all_patterns = map(lambda x: x.strip(), all_patterns.strip().strip(",").split(","))
  for i in xrange(len(all_patterns)):
    all_readers.append(readers.EnsembleReader(
        feature_names=feature_names, feature_sizes=feature_sizes))

  if FLAGS.output_file is "":
    raise ValueError("'output_file' was not specified. "
      "Unable to continue with inference.")

  if FLAGS.input_data_pattern is "":
    raise ValueError("'input_data_pattern' was not specified. "
      "Unable to continue with inference.")

  inference(all_readers, FLAGS.model_checkpoint_path, all_patterns,
    FLAGS.output_file, FLAGS.batch_size, FLAGS.top_k)

  inference(all_readers = all_reader,
            model_checkpoint = FLAGS.model_checkpoint_path,
            all_data_patterns = all_patterns,
            out_file_location = FLAGS.output_file,
            batch_size = FLAGS.batch_size,
            top_k = FLAGS.top_k):

if __name__ == "__main__":
  app.run()
