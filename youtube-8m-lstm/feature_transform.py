from tensorflow import flags

flags.DEFINE_string("feature_transformer", "DefaultTransformer", 
                    "how to preprocess feature, defaults to identical, which means no transform")
flags.DEFINE_string("engineer_types", "identical,avg,std,diff", 
                    "how to preprocess feature, defaults to identical, which means no transform")
flags.DEFINE_integer("time_resolution", 8, 
                    "how many frames to mean pooling at a time")

from all_feature_transform import *
