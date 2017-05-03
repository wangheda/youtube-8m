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

"""Contains model definitions."""
from tensorflow import flags
FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "moe_num_mixtures", 2,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")

flags.DEFINE_integer(
    "deep_chain_layers", 3,
    "The number of layers used for DeepChainModel")
flags.DEFINE_integer(
    "deep_chain_relu_cells", 200,
    "The number of relu cells used for DeepChainModel")
flags.DEFINE_string(
    "deep_chain_relu_type", "relu",
    "The type of relu cells used for DeepChainModel (options are elu and relu)")

flags.DEFINE_bool(
    "deep_chain_use_length", False,
    "The number of relu cells used for DeepChainModel")

flags.DEFINE_integer(
    "hidden_chain_layers", 4,
    "The number of layers used for HiddenChainModel")
flags.DEFINE_integer(
    "hidden_chain_relu_cells", 256,
    "The number of relu cells used for HiddenChainModel")

flags.DEFINE_integer(
    "divergence_model_count", 8,
    "The number of models used in divergence enhancement models")

import sys
from os.path import dirname
if dirname(__file__) not in sys.path:
  sys.path.append(dirname(__file__))
from all_video_models import *
