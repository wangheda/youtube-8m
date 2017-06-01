# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file
# WARNING: This is a temporary script for writing the input directly into files, do not edit base on the file

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
import numpy as np

FLAGS = flags.FLAGS

if __name__ == '__main__':
  flags.DEFINE_string("model_checkpoint_path", "",
                      "The file path to load the model from.")
  flags.DEFINE_string("output_dir", "",
                      "The file to save the predictions to.")
  flags.DEFINE_string(
      "input_data_pattern", "",
      "File glob defining the evaluation dataset in tensorflow.SequenceExample "
      "format. The SequenceExamples are expected to have an 'rgb' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")

  # Model flags.
  flags.DEFINE_bool(
      "frame_features", False,
      "If set, then --eval_data_pattern must be frame-level features. "
      "Otherwise, --eval_data_pattern must be aggregated video-level "
      "features. The model must also be set appropriately (i.e. to read 3D "
      "batches VS 4D batches.")
  flags.DEFINE_integer(
      "batch_size", 8192,
      "How many examples to process per batch.")
  flags.DEFINE_string("feature_names", "mean_rgb", "Name of the feature "
                      "to use for training.")
  flags.DEFINE_string("feature_sizes", "1024", "Length of the feature vectors.")
  flags.DEFINE_integer("file_size", 4096,
                       "Number of frames per batch for DBoF.")

  # Other flags.
  flags.DEFINE_integer("num_readers", 1,
                       "How many threads to use for reading input files.")
  flags.DEFINE_integer("top_k", 20,
                       "How many predictions to output per video.")

def get_input_data_tensors(reader, data_pattern, batch_size, num_readers=1):
  """Creates the section of the graph which reads the input data.

  Args:
    reader: A class which parses the input data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_readers: How many I/O threads to use.

  Returns:
    A tuple containing the features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.

  Raises:
    IOError: If no files matching the given pattern were found.
  """
  with tf.name_scope("input"):
    files = gfile.Glob(data_pattern)
    files.sort()
    if not files:
      raise IOError("Unable to find input files. data_pattern='" +
                    data_pattern + "'")
    logging.info("number of input files: " + str(len(files)))
    filename_queue = tf.train.string_input_producer(
        files, num_epochs=1, shuffle=False)
    examples_and_labels = [reader.prepare_reader(filename_queue)
                           for _ in range(num_readers)]

    video_id_batch, video_batch, unused_labels, num_frames_batch = (
        tf.train.batch_join(examples_and_labels,
                            batch_size=batch_size,
                            allow_smaller_final_batch=True,
                            enqueue_many=True))
    return video_id_batch, video_batch, unused_labels, num_frames_batch

def inference(reader, model_checkpoint_path, data_pattern, out_file_location, batch_size, top_k):
  with tf.Session() as sess:
    video_id_batch, video_batch, video_label_batch, num_frames_batch = get_input_data_tensors(reader, data_pattern, batch_size)

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

    video_id = []
    video_label = []
    video_inputs = []
    video_features = []
    filenum = 0

    directory = FLAGS.output_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        raise IOError("Output path exists! path='" + directory + "'")

    try:
      while not coord.should_stop():
          video_id_batch_val, video_batch_val, video_label_batch_val, num_frames_batch_val = sess.run([video_id_batch, video_batch, video_label_batch, num_frames_batch])
          now = time.time()
          num_examples_processed += len(video_batch_val)

          video_id.append(video_id_batch_val)
          video_label.append(video_label_batch_val)
          video_inputs.append(video_batch_val)

          if num_examples_processed>=FLAGS.file_size:
            assert num_examples_processed==FLAGS.file_size, "num_examples_processed should be equal to file_size"
            video_id = np.concatenate(video_id,axis=0)
            video_label = np.concatenate(video_label,axis=0)
            video_inputs = np.concatenate(video_inputs,axis=0)
            write_to_record(video_id, video_label, video_inputs, filenum, num_examples_processed)
            filenum += 1
            video_id = []
            video_label = []
            video_inputs = []
            num_examples_processed = 0

          logging.info("num examples processed: " + str(num_examples_processed) + " elapsed seconds: " + "{0:.2f}".format(now-start_time))


    except tf.errors.OutOfRangeError:
        logging.info('Done with inference. The output file was written to ' + out_file_location)
    finally:
        coord.request_stop()
        if num_examples_processed<FLAGS.file_size:
            video_id = np.concatenate(video_id,axis=0)
            video_label = np.concatenate(video_label,axis=0)
            video_inputs = np.concatenate(video_inputs,axis=0)
            write_to_record(video_id, video_label, video_inputs, filenum,num_examples_processed)

    coord.join(threads)
    sess.close()

def write_to_record(id_batch, label_batch, input_batch, filenum, num_examples_processed):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_dir + '/' + 'predictions-%04d.tfrecord' % filenum)
    for i in range(num_examples_processed):
        video_id = id_batch[i]
        label = np.nonzero(label_batch[i,:])[0]
        example = get_output_feature(video_id, label, [input_batch[i,:]], ['input'])
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()

def get_output_feature(video_id, labels, features, feature_names):
    feature_maps = {'video_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[video_id])),
                    'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))}
    for feature_index in range(len(feature_names)):
        feature_maps[feature_names[feature_index]] = tf.train.Feature(
            float_list=tf.train.FloatList(value=features[feature_index]))
    example = tf.train.Example(features=tf.train.Features(feature=feature_maps))
    return example

def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)

  # convert feature_names and feature_sizes to lists of values
  feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
      FLAGS.feature_names, FLAGS.feature_sizes)

  if FLAGS.frame_features:
    reader = readers.YT8MFrameFeatureReader(feature_names=feature_names,
                                            feature_sizes=feature_sizes)
  else:
    reader = readers.YT8MAggregatedFeatureReader(feature_names=feature_names,
                                                 feature_sizes=feature_sizes)

  if FLAGS.output_dir is "":
    raise ValueError("'output_dir' was not specified. "
      "Unable to continue with inference.")

  if FLAGS.input_data_pattern is "":
    raise ValueError("'input_data_pattern' was not specified. "
      "Unable to continue with inference.")

  inference(reader, FLAGS.model_checkpoint_path, FLAGS.input_data_pattern,
      FLAGS.output_dir, FLAGS.batch_size, FLAGS.top_k)


if __name__ == "__main__":
  app.run()
