from abc import ABC, abstractmethod
from functools import reduce
from operator import mul
from math import ceil
from typing import Iterable, Optional, Union, Sequence, Type

import numpy as np
import torch
from torch.nn import Module, Parameter

from .nodes import Nodes
from .topology import AbstractConnection


class ConnectedNodes(ABC, Module):
    # language=rst
    """
    [summary]
    """

    def __init__(
        self,
        nodes_type: Type[Nodes],
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        [summary]
        """
        super().__init__()
        self.nodes_type = nodes_type
        self.n = n
        self.shape = shape

        b = kwargs.get("b", None)
        if b is not None:
            self.b = Parameter(b, requires_grad=False)
        else:
            self.b = None

    @abstractmethod
    @property 
    def Nodes(self) -> Nodes:
        pass

    @abstractmethod
    @property 
    def Connection(self) -> AbstractConnection:
        pass


class RandomConnections(ConnectedNodes):
    def __init__(
        self,
        nodes_type: Type[Nodes],
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        [summary]
        """
        super().__init__(nodes_type, n, shape, kwargs)

    @property 
    def Nodes(self) -> Nodes:
        if self.wmin == -np.inf or self.wmax == np.inf:
            w = torch.clamp(torch.rand(self.n, self.n), self.wmin, self.wmax)
        else:
            w = self.wmin + torch.rand(self.n, self.n) * (self.wmax - self.wmin)
        w = torch.clamp(torch.as_tensor(w), self.wmin, self.wmax)
        w[np.diag_indices(w.shape[0])] = 0.

        mask = (torch.rand(self.n, self.n) + self.connection_chance).int().bool()
        exc_inh = torch.tensor([1]* int(self.excitatory_ratio * self.n) +\
             [-1]* (ceil((1-self.excitatory_ratio) * self.n)))
        w *= mask * exc_inh

        self.w = Parameter(w, requires_grad=False)

        return self.nodes_type(
            self.n,
            self.shape,
            self.traces,
            self.traces_additive,
            self.tc_trace,
            self.trace_scale,
            self.sum_input,
            self.learning,
            )
