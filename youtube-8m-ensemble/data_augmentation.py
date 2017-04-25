from tensorflow import flags

flags.DEFINE_string("data_augmenter", "DefaultAugmenter", 
                    "how to preprocess feature, defaults to identical, which means no transform")
flags.DEFINE_float("input_noise_level", 0.2, 
                    "the standard deviation of normal noise added to input")

from all_data_augmentation import *
