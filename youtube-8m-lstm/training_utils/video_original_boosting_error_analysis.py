import random
from datetime import datetime
import tensorflow as tf
from tensorflow import flags
FLAGS = flags.FLAGS

if __name__=="__main__":
  flags.DEFINE_string("train_labels_file", "", "The file in which every line is a video_id and the corresponding labels.")
  flags.DEFINE_string("video_id_file", "", "The file in which every line is a video_id.")
  flags.DEFINE_string("reweight_freq_file", "", "The file in which every line is a weight used in boosting.")
  flags.DEFINE_string("label_names_file", "", "The file containing the names of labels.")
  flags.DEFINE_string("output_analysis_file", "", "The file containing the weight, url and labels.")

if __name__=="__main__":
  video_ids = [line.strip() for line in open(FLAGS.video_id_file) if len(line.strip()) > 0]
  video_freqs = [float(line.strip()) for line in open(FLAGS.reweight_freq_file) if len(line.strip()) > 0]

  names_dict = dict([line.strip().split(",") for line in open(FLAGS.label_names_file)])
  video_name_dict = {}
  label_dict = {}
  with open(FLAGS.train_labels_file) as fi:
    fi.next()
    for line in fi:
      if len(line.strip()) > 0:
        words = line.strip().split(",")
        if len(words) == 2:
          video_id, label_ids = words
          label_names = " / ".join(map(lambda l: names_dict.get(l, "OOV"), label_ids.split()))
          video_name_dict[video_id] = label_names

  freq_id_pairs = sorted(zip(video_freqs, video_ids), reverse=True)

  with open(FLAGS.output_analysis_file, "w") as fo:
    for v_freq, v_id in freq_id_pairs:
      print >> fo, v_freq, "\t", "https://www.youtube.com/watch?v=%s"%v_id, "\t", video_name_dict.get(v_id, "OOV")

