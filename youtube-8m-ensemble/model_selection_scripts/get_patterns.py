import sys
from tensorflow import flags

FLAGS = flags.FLAGS

if __name__=="__main__":
  flags.DEFINE_string("train_path", "", "The directory where training files locates.")
  flags.DEFINE_string("candidates", "", "The candidate methods.")

if __name__=="__main__":
  candidate_methods = map(lambda x: x.strip(), FLAGS.candidates.strip().split(","))
  train_path = FLAGS.train_path
  output_path = ",".join(map(lambda x: "%s/%s/*.tfrecord"%(train_path, x), candidate_methods))
  sys.stdout.write(output_path)
  sys.stdout.flush()
