# Copyright 2020 Google LLC
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

import gin

from abc import ABC, abstractmethod

class GeneratorBeamHandler(ABC):

  # Write abstract functions that return instantiated beam classes.
  @abstractmethod
  def GetSampleDoFn(self):
    pass

  @abstractmethod
  def GetWriteDoFn(self):
    pass

  @abstractmethod
  def GetConvertParDo(self):
    pass

  @abstractmethod
  # This should eventually take a gin-specified tuple of GNN classes.
  def GetBenchmarkParDo(self):
    pass

  @abstractmethod
  def GetGraphMetricsParDo(self):
    pass

  @abstractmethod
  def SetOutputPath(self, output_path):
    pass

@gin.configurable
class GeneratorBeamHandlerWrapper:

  @gin.configurable
  def __init__(self, handler, nsamples):
    self.nsamples = nsamples
    self.handler = handler

  def SetOutputPath(self, output_path):
    self.output_path = output_path
    self.handler.SetOutputPath(output_path)