
import tensorflow as tf

class IdenticalTransformer:
  def transform(self, model_input_raw, num_frames, **unused_params):
    return model_input_raw, num_frames
