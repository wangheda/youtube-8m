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
  flags.DEFINE_string("model_checkpoint_path", "",
                      "The file to load the model files from. ")
  flags.DEFINE_string("train_dir", "/tmp/yt8m/",
                      "The directory to write the result in. ")
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
  flags.DEFINE_string(
      "model", "LinearRegressionModel",
      "Which architecture to use for the model.")
  flags.DEFINE_integer("batch_size", 256,
                       "How many examples to process per batch.")
  flags.DEFINE_string("label_loss", "CrossEntropyLoss",
                      "Loss computed on validation data")

  # Other flags.
  flags.DEFINE_boolean("run_once", True, "Whether to run eval only once.")
  flags.DEFINE_boolean("echo_gap", False, "Whether to echo GAP at the end.")
  flags.DEFINE_integer("top_k", 20, "How many predictions to output per video.")

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
                all_eval_data_patterns,
                input_data_pattern,
                model,
                label_loss_fn,
                batch_size=256):
  """Creates the Tensorflow graph for evaluation.

  Args:
    all_readers: The data file reader. It should inherit from BaseReader.
    model: The core model (e.g. logistic or neural net). It should inherit
           from BaseModel.
    all_eval_data_patterns: glob path to the evaluation data files.
    label_loss_fn: What kind of loss to apply to the model. It should inherit
                from BaseLoss.
    batch_size: How many examples to process at a time.
  """

  global_step = tf.Variable(0, trainable=False, name="global_step")

  model_input_raw_tensors = []
  labels_batch_tensor = None
  video_id_batch = None
  for reader, data_pattern in zip(all_readers, all_eval_data_patterns):
    unused_video_id, model_input_raw, labels_batch, unused_num_frames = (
        get_input_evaluation_tensors(
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
        get_input_evaluation_tensors(
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
    tf.summary.histogram("model_activations", predictions)
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
  tf.add_to_collection("summary_op", tf.summary.merge_all())


def evaluation_loop(video_id_batch, prediction_batch, label_batch, loss,
                    summary_op, saver, summary_writer, evl_metrics,
                    last_global_step_val):
  """Run the evaluation loop once.

  Args:
    video_id_batch: a tensor of video ids mini-batch.
    prediction_batch: a tensor of predictions mini-batch.
    label_batch: a tensor of label_batch mini-batch.
    loss: a tensor of loss for the examples in the mini-batch.
    summary_op: a tensor which runs the tensorboard summary operations.
    saver: a tensorflow saver to restore the model.
    summary_writer: a tensorflow summary_writer
    evl_metrics: an EvaluationMetrics object.
    last_global_step_val: the global step used in the previous evaluation.

  Returns:
    The global_step used in the latest model.
  """

  global_step_val = -1
  with tf.Session() as sess:
    checkpoint = FLAGS.model_checkpoint_path
    if checkpoint:
      logging.info("Loading checkpoint for eval: " + checkpoint)
      # Restores from checkpoint
      saver.restore(sess, checkpoint)
      # Assuming model_checkpoint_path looks something like:
      # /my-favorite-path/yt8m_train/model.ckpt-0, extract global_step from it.
      global_step_val = checkpoint.split("/")[-1].split("-")[-1]
    else:
      logging.info("No checkpoint file found.")
      return global_step_val

    sess.run([tf.local_variables_initializer()])

    # Start the queue runners.
    fetches = [video_id_batch, prediction_batch, label_batch, loss, summary_op]
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(
            sess, coord=coord, daemon=True,
            start=True))
      logging.info("enter eval_once loop global_step_val = %s. ",
                   global_step_val)

      evl_metrics.clear()

      examples_processed = 0
      while not coord.should_stop():
        batch_start_time = time.time()

        _, predictions_val, labels_val, loss_val, summary_val = sess.run(fetches)

        seconds_per_batch = time.time() - batch_start_time
        example_per_second = labels_val.shape[0] / seconds_per_batch
        examples_processed += labels_val.shape[0]

        iteration_info_dict = evl_metrics.accumulate(predictions_val,
                                                     labels_val, loss_val)
        iteration_info_dict["examples_per_second"] = example_per_second

        iterinfo = utils.AddGlobalStepSummary(
            summary_writer,
            global_step_val,
            iteration_info_dict,
            summary_scope="Eval")
        logging.info("examples_processed: %d | %s", examples_processed,
                     iterinfo)

    except tf.errors.OutOfRangeError as e:
      logging.info(
          "Done with batched inference. Now calculating global performance "
          "metrics.")
      # calculate the metrics for the entire epoch
      epoch_info_dict = evl_metrics.get()
      epoch_info_dict["epoch_id"] = global_step_val

      summary_writer.add_summary(summary_val, global_step_val)
      epochinfo = utils.AddEpochSummary(
          summary_writer,
          global_step_val,
          epoch_info_dict,
          summary_scope="Eval")
      if FLAGS.echo_gap:
        print "GAP =", epoch_info_dict["gap"]
      logging.info(epochinfo)
      evl_metrics.clear()
    except Exception as e:  # pylint: disable=broad-except
      logging.info("Unexpected exception: " + str(e))
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

    return global_step_val


def evaluate():
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
          feature_names=["input"], feature_sizes=[1024+128])
      input_data_pattern = FLAGS.input_data_pattern

    # find the model
    model = find_class_by_name(FLAGS.model, [ensemble_level_models])()
    label_loss_fn = find_class_by_name(FLAGS.label_loss, [losses])()

    if FLAGS.eval_data_patterns is "":
      raise IOError("'eval_data_patterns' was not specified. " +
                     "Nothing to evaluate.")

    build_graph(
        all_readers=all_readers,
        input_reader=input_reader,
        all_eval_data_patterns=all_patterns,
        input_data_pattern=input_data_pattern,
        model=model,
        label_loss_fn=label_loss_fn,
        batch_size=FLAGS.batch_size)
    logging.info("built evaluation graph")
    video_id_batch = tf.get_collection("video_id_batch")[0]
    prediction_batch = tf.get_collection("predictions")[0]
    label_batch = tf.get_collection("labels")[0]
    loss = tf.get_collection("loss")[0]
    summary_op = tf.get_collection("summary_op")[0]

    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(
        FLAGS.train_dir, graph=tf.get_default_graph())

    evl_metrics = eval_util.EvaluationMetrics(FLAGS.num_classes, FLAGS.top_k)

    last_global_step_val = -1
    last_global_step_val = evaluation_loop(video_id_batch, prediction_batch,
                                           label_batch, loss, summary_op,
                                           saver, summary_writer, evl_metrics,
                                           last_global_step_val)


def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)
  print("tensorflow version: %s" % tf.__version__)
  evaluate()


if __name__ == "__main__":
  app.run()

