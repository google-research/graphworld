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