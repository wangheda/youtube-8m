import random
from datetime import datetime
import tensorflow as tf
from tensorflow import flags
FLAGS = flags.FLAGS

if __name__=="__main__":
  flags.DEFINE_string("video_id_file", "", "The file in which every line is a video_id.")
  flags.DEFINE_string("output_freq_file", "", "Output the corresponding freq of video_ids.")

if __name__=="__main__":
  with open(FLAGS.video_id_file) as F:
    word_list = []
    freq_dict = {}

    # OOV
    line = F.next()

    for line in F:
      word = line.strip()
      if word:
        word_list.append(word)
        freq_dict[word] = 0

    # random sample
    random.seed(datetime.now())
    for i in xrange(len(word_list)):
      index = random.randint(0,len(word_list)-1)
      word = word_list[index]
      freq_dict[word] += 1

    # get weight
    word_weights = []
    word_weights.append(1)
    for word in word_list:
      word_weights.append(freq_dict[word])

    # output weight
    with open(FLAGS.output_freq_file, "w") as out:
      out.writelines([str(item)+"\n" for item in word_weights])
