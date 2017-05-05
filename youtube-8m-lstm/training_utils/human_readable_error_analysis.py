import random
from datetime import datetime
import tensorflow as tf
from tensorflow import flags
FLAGS = flags.FLAGS

if __name__=="__main__":
  flags.DEFINE_string("input_error_analysis_file", "", "The file in which every line is a video_id.")
  flags.DEFINE_string("output_error_analysis_file", "", "The file in which every line is a video_id.")
  flags.DEFINE_string("label_names_file", "", "The file containing the names of labels.")
  flags.DEFINE_integer("top_k", 20, "How many predictions are in each line.")

def csv_line(str_list):
  return ",".join(map(lambda s: "\""+s+"\"", str_list)) + "\n"
  
if __name__=="__main__":
  top_k = FLAGS.top_k
  label_names = dict([line.strip().split(",") for line in open(FLAGS.label_names_file)])
  with open(FLAGS.input_error_analysis_file) as fi:
    # get rid of the first line
    fi.next()
    with open(FLAGS.output_error_analysis_file, "w") as fo: 
      fo.write(csv_line(["Cause", "URL", "ERROR", "field", "list"]))
      words_list = []

      for line in fi:
        words = line.strip().split("\t")
        words_list.append(words)
      words_list.sort(key=lambda x: -float(x[1]))

      for words in words_list:
        empty = ""
        cause = ""
        url = words[0]
        error = words[1]
        predictions = words[2:2+2*top_k]
        p_indices = predictions[0::2]
        p_words = map(lambda w: label_names.get(w, "OOV"), p_indices)
        p_values = predictions[1::2]
        labels = words[2+2*top_k:]
        l_indices = labels[0::2]
        l_words = map(lambda w: label_names.get(w, "OOV"), l_indices)
        l_values = labels[1::2]
        fo.write(csv_line([empty, url, error, "predictions"] + p_words))
        fo.write(csv_line([empty, empty, empty, empty] + p_values))
        fo.write(csv_line([empty, empty, empty, "labels"] + l_words))
        fo.write(csv_line([empty, empty, empty, empty] + l_values))
