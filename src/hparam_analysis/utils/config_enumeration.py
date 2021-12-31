from itertools import product
from typing import Optional, List

import gin

@gin.configurable
def enumerate_configs(
    hidden_channel_values: List[int] = [16],
    weight_decay_values: List[float] = [5e-5],
    dropout_values: List[float] = [0.5],
    learning_rate_values: List[float] = [1e-2]
  ):

  configs = list(product(*[
    hidden_channel_values,
    weight_decay_values,
    dropout_values,
    learning_rate_values
  ]))

  configs = [
    {'index': i,
     'hidden_channels': a,
     'weight_decay': b,
     'dropout': c,
     'learning_rate': d} for
    i, (a, b, c, d) in enumerate(configs)
  ]

  return configs


