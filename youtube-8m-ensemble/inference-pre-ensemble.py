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
import numpy as np

import tensorflow as tf
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging

import utils
import eval_util
import losses
import readers
import ensemble_level_models

FLAGS = flags.FLAGS

if __name__ == '__main__':
  flags.DEFINE_string("model_checkpoint_path", None,
                      "The file path to load the model from.")
  flags.DEFINE_string("train_dir", "",
                      "The directory to load the model from.")
  flags.DEFINE_string("output_dir", "",
                      "The file to save the predictions to.")
  flags.DEFINE_string(
      "input_data_patterns", "",
      "File globs defining the evaluation dataset in tensorflow.SequenceExample format.")
  flags.DEFINE_string(
      "input_data_pattern", None,
      "File globs for original model input.")
  flags.DEFINE_string("feature_names", "predictions", "Name of the feature "
                      "to use for training.")
  flags.DEFINE_string("feature_sizes", "4716", "Length of the feature vectors.")

  # Model flags.
  flags.DEFINE_string(
      "model", "MeanModel",
      "Which architecture to use for the model.")
  flags.DEFINE_integer("batch_size", 256,
                       "How many examples to process per batch.")
  flags.DEFINE_string("label_loss", "CrossEntropyLoss",
                      "Loss computed on validation data")
  flags.DEFINE_integer("file_size", 4096,
                       "Number of frames per batch for DBoF.")

def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)

def get_input_data_tensors(reader,
                                 data_pattern,
                                 batch_size=256):
  logging.info("Using batch size of " + str(batch_size) + " for evaluation.")
  with tf.name_scope("eval_input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find the evaluation files.")
    logging.info("number of evaluation files: " + str(len(files)))
    files.sort()
    filename_queue = tf.train.string_input_producer(
        files, shuffle=False, num_epochs=1)
    eval_data = reader.prepare_reader(filename_queue)
    return tf.train.batch(
        eval_data,
        batch_size=batch_size,
        capacity=4 * batch_size,
        allow_smaller_final_batch=True,
        enqueue_many=True)

def build_graph(all_readers,
                all_data_patterns,
                input_reader,
                input_data_pattern,
                model,
                label_loss_fn,
                batch_size=256):
  """Creates the Tensorflow graph for evaluation.

  Args:
    all_readers: The data file reader. It should inherit from BaseReader.
    model: The core model (e.g. logistic or neural net). It should inherit
           from BaseModel.
    all_data_patterns: glob path to the evaluation data files.
    label_loss_fn: What kind of loss to apply to the model. It should inherit
                from BaseLoss.
    batch_size: How many examples to process at a time.
  """

  global_step = tf.Variable(0, trainable=False, name="global_step")

  model_input_raw_tensors = []
  labels_batch_tensor = None
  video_id_batch = None
  for reader, data_pattern in zip(all_readers, all_data_patterns):
    unused_video_id, model_input_raw, labels_batch, unused_num_frames = (
        get_input_data_tensors(
            reader,
            data_pattern,
            batch_size=batch_size))
    if labels_batch_tensor is None:
      labels_batch_tensor = labels_batch
    if video_id_batch is None:
      video_id_batch = unused_video_id
    model_input_raw_tensors.append(tf.expand_dims(model_input_raw, axis=2))

  original_input = None
  if input_data_pattern is not None:
    unused_video_id, original_input, unused_labels_batch, unused_num_frames = (
        get_input_data_tensors(
            input_reader,
            input_data_pattern,
            batch_size=batch_size))

  model_input = tf.concat(model_input_raw_tensors, axis=2)
  labels_batch = labels_batch_tensor

  with tf.name_scope("model"):
    result = model.create_model(model_input,
                                labels=labels_batch,
                                vocab_size=reader.num_classes,
                                original_input=original_input,
                                is_training=False)
    predictions = result["predictions"]
    if "loss" in result.keys():
      label_loss = result["loss"]
    else:
      label_loss = label_loss_fn.calculate_loss(predictions, labels_batch)

  tf.add_to_collection("global_step", global_step)
  tf.add_to_collection("loss", label_loss)
  tf.add_to_collection("predictions", predictions)
  tf.add_to_collection("input_batch", model_input)
  tf.add_to_collection("video_id_batch", video_id_batch)
  tf.add_to_collection("labels", tf.cast(labels_batch, tf.float32))

def inference_loop(video_id_batch, prediction_batch,
                   label_batch, saver,
                   output_dir, batch_size):
  with tf.Session() as sess:
    checkpoint = FLAGS.model_checkpoint_path
    if checkpoint is None:
      checkpoint = tf.train.latest_checkpoint(FLAGS.train_dir)
    if checkpoint:
      logging.info("Loading checkpoint for eval: " + checkpoint)
      saver.restore(sess, checkpoint)
      global_step_val = checkpoint.split("/")[-1].split("-")[-1]
    else:
      logging.info("No checkpoint file found.")
      return global_step_val

    sess.run([tf.local_variables_initializer()])

    # Start the queue runners.
    fetches = [video_id_batch, prediction_batch, label_batch]
    coord = tf.train.Coordinator()
    start_time = time.time()

    video_ids = []
    video_labels = []
    video_features = []
    filenum = 0
    num_examples_processed = 0
    total_num_examples_processed = 0

    directory = FLAGS.output_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        raise IOError("Output path exists! path='" + directory + "'")

    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(
            sess, coord=coord, daemon=True,
            start=True))
      logging.info("enter eval_once loop global_step_val = %s. ",
                   global_step_val)

      while not coord.should_stop():
        ids_val, predictions_val, labels_val = None, None, None

        ids_val, predictions_val, labels_val = sess.run(fetches)

        video_ids.append(ids_val)
        video_labels.append(labels_val)
        video_features.append(predictions_val)
        num_examples_processed += len(ids_val)

        ids_val, predictions_val, labels_val = None, None, None

        if num_examples_processed >= FLAGS.file_size:
          assert num_examples_processed==FLAGS.file_size, "num_examples_processed should be equal to %d"%FLAGS.file_size
          video_ids = np.concatenate(video_ids, axis=0)
          video_labels = np.concatenate(video_labels, axis=0)
          video_features = np.concatenate(video_features, axis=0)
          write_to_record(video_ids, video_labels, video_features, filenum, num_examples_processed)

          video_ids = []
          video_labels = []
          video_features = []
          filenum += 1
          total_num_examples_processed += num_examples_processed

          now = time.time()
          logging.info("num examples processed: " + str(num_examples_processed) + " elapsed seconds: " + "{0:.2f}".format(now-start_time))
          num_examples_processed = 0

    except tf.errors.OutOfRangeError as e:
      if ids_val is not None:
        video_ids.append(ids_val)
        video_labels.append(labels_val)
        video_features.append(predictions_val)
        num_examples_processed += len(ids_val)

      if 0 < num_examples_processed <= FLAGS.file_size:
        video_ids = np.concatenate(video_ids, axis=0)
        video_labels = np.concatenate(video_labels, axis=0)
        video_features = np.concatenate(video_features, axis=0)
        write_to_record(video_ids, video_labels, video_features, filenum, num_examples_processed)
        total_num_examples_processed += num_examples_processed

        now = time.time()
        logging.info("num examples processed: " + str(num_examples_processed) + " elapsed seconds: " + "{0:.2f}".format(now-start_time))
        num_examples_processed = 0

      logging.info("Done with inference. %d samples was written to %s" % (total_num_examples_processed, FLAGS.output_dir))
    except Exception as e:  # pylint: disable=broad-except
      logging.info("Unexpected exception: " + str(e))
    finally:
      coord.request_stop()

    coord.join(threads, stop_grace_period_secs=10)


def write_to_record(video_ids, video_labels, video_features, filenum, num_examples_processed):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_dir + '/' + 'predictions-%04d.tfrecord' % filenum)
    for i in range(num_examples_processed):
        video_id = video_ids[i]
        video_label = np.nonzero(video_labels[i,:])[0]
        example = get_output_feature(video_id, video_label, [video_features[i,:]], ['predictions'])
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()

def get_output_feature(video_id, video_label, video_feature, feature_names):
    feature_maps = {'video_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[video_id])),
                    'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=video_label))}
    for feature_index in range(len(feature_names)):
        feature_maps[feature_names[feature_index]] = tf.train.Feature(
            float_list=tf.train.FloatList(value=video_feature[feature_index]))
    example = tf.train.Example(features=tf.train.Features(feature=feature_maps))
    return example

def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default():
    # convert feature_names and feature_sizes to lists of values
    feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
        FLAGS.feature_names, FLAGS.feature_sizes)
  
    # prepare a reader for each single model prediction result
    all_readers = []
  
    all_patterns = FLAGS.input_data_patterns
    all_patterns = map(lambda x: x.strip(), all_patterns.strip().strip(",").split(","))
    for i in xrange(len(all_patterns)):
      reader = readers.EnsembleReader(
          feature_names=feature_names, feature_sizes=feature_sizes)
      all_readers.append(reader)

    input_reader = None
    input_data_pattern = None
    if FLAGS.input_data_pattern is not None:
      input_reader = readers.EnsembleReader(
          feature_names=["input"], feature_sizes=[1024+128])
      input_data_pattern = FLAGS.input_data_pattern

    model = find_class_by_name(FLAGS.model, [ensemble_level_models])()
    label_loss_fn = find_class_by_name(FLAGS.label_loss, [losses])()

    if FLAGS.input_data_patterns is "":
      raise IOError("'input_data_patterns' was not specified. " +
                     "Nothing to evaluate.")

    build_graph(
        all_readers=all_readers,
        input_reader=input_reader,
        all_data_patterns=all_patterns,
        input_data_pattern=input_data_pattern,
        model=model,
        label_loss_fn=label_loss_fn,
        batch_size=FLAGS.batch_size)

    logging.info("built evaluation graph")
    video_id_batch = tf.get_collection("video_id_batch")[0]
    prediction_batch = tf.get_collection("predictions")[0]
    label_batch = tf.get_collection("labels")[0]

    saver = tf.train.Saver(tf.global_variables())

    inference_loop(video_id_batch, prediction_batch,
                   label_batch, saver, 
                   FLAGS.output_dir, FLAGS.batch_size)
  
if __name__ == "__main__":
  app.run()
