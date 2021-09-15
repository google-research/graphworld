# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from sklearn.metrics import mean_squared_error

def MseWrapper(pred, true, scale=False):
  """Wrapper for Mean-Squared-Error eval metric.

  Arguments:
    pred: iterable of predicted values
    true: iterable of true values
    scale: if True, return value is divided by var(true)
  """
  mse = mean_squared_error(pred, true)
  if scale:
    mse /= np.var(true)
  return mse
