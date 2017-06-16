# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains a collection of models which operate on variable-length sequences.
"""
from tensorflow import flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("iterations", 30,
                     "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")
flags.DEFINE_integer("dbof_cluster_size", 8192,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 1024,
                     "Number of units in the DBoF hidden layer.")
flags.DEFINE_string("dbof_pooling_method", "max",
                    "The pooling method used in the DBoF cluster layer. "
                    "Choices are 'average' and 'max'.")
flags.DEFINE_string("video_level_classifier_model", "MoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")
flags.DEFINE_string("video_level_classifier_support_model", "MoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")
flags.DEFINE_bool("rnn_swap_memory", False, "If true, swap_memory = True.")
flags.DEFINE_string("lstm_cells", "1024", "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")
flags.DEFINE_integer("gru_cells", 1024, "Number of GRU cells.")
flags.DEFINE_integer("gru_layers", 2, "Number of GRU layers.")

flags.DEFINE_integer("attention_size", 1, "Number of attention layers.")
flags.DEFINE_integer("num_attentions", 5, "Number of attention cells per layer.")

flags.DEFINE_string("cnn_filter_sizes", "1,2,3", "Sizes of cnn filters.")
flags.DEFINE_string("cnn_filter_nums", "256,256,256", "Numbers of every cnn filters.")
flags.DEFINE_integer("cnn_pooling_k", 4, "The k value for max-k pooling.")

flags.DEFINE_string("lstm_normalization", "identical",
                    "which normalization method")

flags.DEFINE_integer("mm_label_embedding", 256,
                    "size of label embedding vector")

flags.DEFINE_string("wide_and_deep_models", "FrameLevelLogisticModel,LstmMemoryModel",
                    "size of label embedding vector")

flags.DEFINE_integer("deep_cnn_base_size", 128,
                     "basic cnn size")

flags.DEFINE_integer("lstm_look_back", 3,
                     "how many adjacent input for a cell to look at")

flags.DEFINE_integer("lstm_attentions", 8, "Attention size in lstm_attention_max_pooling_model.")

flags.DEFINE_integer("positional_embedding_size", 32, "Positional embedding dimension use in lstm_positional_attention_max_pooling_model.")

flags.DEFINE_bool("is_training", False, "used in batch normalization.")

flags.DEFINE_integer("multiscale_cnn_lstm_layers", 1, "number of layers in multiscale cnn_lstm.")

flags.DEFINE_integer("frame_seg_relu_cells", 256, "number of relu cells in frame-seg model.")

flags.DEFINE_integer("distillchain_relu_cells", 256, "number of relu cells in distillchain model.")

flags.DEFINE_integer("cnn_num_filters", 512, "number of filters in cnn conv layer.")

import sys
from os.path import dirname
if dirname(__file__) not in sys.path:
  sys.path.append(dirname(__file__))
from all_frame_models import *
