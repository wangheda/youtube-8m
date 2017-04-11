from tensorflow import flags

flags.DEFINE_string("data_augmenter", "DefaultAugmenter", 
                    "how to preprocess feature, defaults to identical, which means no transform")

from all_data_augmentation import *
