import random
from datetime import datetime
import tensorflow as tf
from tensorflow import flags
FLAGS = flags.FLAGS

if __name__=="__main__":
  flags.DEFINE_string("main_conf_file", "", "The conf file to sample from.")
  flags.DEFINE_string("sub_conf_file", "", "The conf file to randomly generate.")

if __name__=="__main__":
  with open(FLAGS.main_conf_file) as F:
    methods = []
    sample_methods = []

    # methods
    for line in F:
      m = line.strip()
      if m:
        methods.append(m)

    # random sample
    random.seed(datetime.now())
    for i in xrange(len(methods)):
      index = random.randint(0,len(methods)-1)
      m = methods[index]
      sample_methods.append(m)

    # output weight
    with open(FLAGS.sub_conf_file, "w") as out:
      out.writelines([m+"\n" for m in sample_methods])
