import sys
from tensorflow import flags

FLAGS = flags.FLAGS

if __name__=="__main__":
  flags.DEFINE_string("log_file", "", "The file that log models performances.")
  flags.DEFINE_string("sorted_log_file", "", "The file that log models performances (sorted by GAP).")
  flags.DEFINE_integer("top_k", 10, "The number of top models reserved.")

if __name__=="__main__":
  log_file = FLAGS.log_file
  with open(log_file) as F:
    lines = F.readlines()
    models = map(lambda x: x.strip(), lines[::2])
    perfs = map(lambda x: float(x.strip().split("=")[-1]), lines[1::2])
    perfs = perfs[:len(models)]
    model_perfs = sorted(zip(perfs, models), reverse=True)

    with open(FLAGS.sorted_log_file, "w") as Fo:
      Fo.writelines(["%f\t%s\n"%(x,y) for x,y in model_perfs])

    for perf, model in model_perfs[:FLAGS.top_k]:
      print model
      

