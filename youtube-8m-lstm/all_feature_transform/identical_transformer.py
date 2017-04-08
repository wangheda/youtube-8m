
import tensorflow as tf

class IdenticalTransformer:
  def transform(self, model_input_raw, **unused_params):
    return model_input_raw
