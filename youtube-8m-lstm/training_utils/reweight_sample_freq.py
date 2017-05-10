import random
import math
from datetime import datetime
import tensorflow as tf
from tensorflow import flags
from collections import defaultdict

FLAGS = flags.FLAGS

if __name__=="__main__":
  flags.DEFINE_string("video_id_file", "", "The file in which every line is a video_id.")
  flags.DEFINE_string("input_freq_file", "", "The previous weight of each video.")
  flags.DEFINE_string("input_error_file", "", "The error of each video.")
  flags.DEFINE_string("output_freq_file", "", "Output the corresponding freq of video_ids.")
  flags.DEFINE_float("clip_weight", None, "The max value of sample weight. "
                     "Exceeding part will be randomly distributed to other videos.")
  flags.DEFINE_float("discard_weight", None, "The max value of sample weight. "
                     "The weight exceeding it will be cut to zero with its weight "
                     "randomly distributed to other videos.")

if __name__=="__main__":
  word_list = []
  freq_dict = defaultdict(lambda: 1.0)
  error_dict = defaultdict(float)

  with open(FLAGS.video_id_file) as F:
    # read in video_ids
    for line in F:
      word = line.strip()
      if word:
        word_list.append(word)

  with open(FLAGS.input_freq_file) as F:
    # read in frequencies
    i = 0
    for line in F:
      word = word_list[i]
      freq = float(line.strip())
      freq_dict[word] = freq
      i += 1

  with open(FLAGS.input_error_file) as F:
    # read in errors
    for line in F:
      words = line.strip().split()
      if len(words) == 2:
        video_id, error = words
        error = float(error)
        error_dict[video_id] = error

  # compute the new reweight value
  epsilon = 1e-6
  global_error_rate = sum(error_dict.values()) / len(error_dict)
  ratio = math.log((1.0 + epsilon - global_error_rate) / (global_error_rate + epsilon))
  for video_id in word_list:
    freq_dict[video_id] = freq_dict[video_id] * math.exp(ratio * error_dict[video_id])

  pool = 0.0

  # discarding weight
  if FLAGS.discard_weight:
    # discarding
    discard = FLAGS.discard_weight
    pool = 0.0
    for video_id in word_list:
      if freq_dict[video_id] > discard:
        pool += freq_dict[video_id]
        freq_dict[video_id] = 0.0

  # clipping weight
  if FLAGS.clip_weight:
    # clipping 
    clip = FLAGS.clip_weight
    pool = 0.0
    for video_id in word_list:
      if freq_dict[video_id] > clip:
        pool += freq_dict[video_id] - clip
        freq_dict[video_id] = clip

  if pool > 0:
    # re-distributed to other video_id
    random.seed(datetime.now())
    avg_pool = pool / len(word_list)
    for video_id in word_list:
      if freq_dict[video_id] > 0:
        freq_dict[video_id] += random.random() * 2 * avg_pool

  # make the average value 1.0
  freq_rel_ratio = max(sum(freq_dict.values()) / len(freq_dict), 1e-6)
  for video_id in word_list:
    freq_dict[video_id] /= freq_rel_ratio

  print "average value in freq_dict =", sum(freq_dict.values()) / len(freq_dict)

  # write to output_freq_file
  with open(FLAGS.output_freq_file, "w") as F:
    lines_to_write = ["%f\n" % freq_dict[video_id] for video_id in word_list]
    F.writelines(lines_to_write)


      



