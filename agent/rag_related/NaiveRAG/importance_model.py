import abc
from collections.abc import Callable, Iterable, Sequence
import datetime
import threading
import numpy as np
import pandas as pd

class ImportanceModel(metaclass=abc.ABCMeta):
  """Memory importance module for generative agents."""

  @abc.abstractmethod
  def importance(self,
                 memory: str,
                 context: Sequence[tuple[str, float]] = ()) -> float:
    """Computes importance of a memory.

    Args:
      memory: a memory (text) to compute importance of
      context: a sequence of tuples of (old memory (str), importance
        (float between 0 and 1)) used to provide context and relative scale for
        the decision of the importance of the new memory.

    Returns:
      Value of importance in the [0,1] interval
    """

    raise NotImplementedError

class ConstantImportanceModel(ImportanceModel):
  """Memory importance function that always returns a constant.

  This is useful for debugging since it doesn't call LLM.
  """

  def __init__(
      self,
      fixed_importance: float = 1.0,
  ):
    """Initialises an instance.

    Args:
      fixed_importance: the constant to return
    """
    self._fixed_importance = fixed_importance

  def importance(self,
                 memory: str,
                 context: Sequence[tuple[str, float]] = ()) -> float:
    """Computes importance of a memory by querying the LLM.

    Args:
      memory: memory to compute importance of
      context: unused

    Returns:
      Value of importance
    """
    del memory, context

    return self._fixed_importance
