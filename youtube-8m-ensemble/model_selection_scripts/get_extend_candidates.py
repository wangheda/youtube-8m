import sys
from tensorflow import flags

FLAGS = flags.FLAGS

if __name__=="__main__":
  flags.DEFINE_string("top_k_file", "", "The file that contains the top-k ensemble models.")
  flags.DEFINE_string("all_models_conf", "", "The file that contains all available single models.")

if __name__=="__main__":
  all_models = [line.strip() for line in open(FLAGS.all_models_conf) if len(line.strip()) > 0]
  extend_candidates = set()
  with open(FLAGS.top_k_file) as F:
    ensemble_models = [line.strip().split(",") for line in F.readlines() if len(line.strip()) > 0]
    for em in ensemble_models:
      for model in all_models:
        if model not in em:
          new_combination = ",".join(sorted(em + [model]))
          if new_combination not in extend_candidates:
            extend_candidates.add(new_combination)
  for candidate in extend_candidates:
    print candidate

