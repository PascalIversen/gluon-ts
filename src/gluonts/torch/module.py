from functools import partial

from gluonts.core.component import DType
from gluonts.generic_network import GenericNetwork
import mxnet as mx
from pathlib import Path
import numpy as np

from typing import Optional, Callable
import torch
from gluonts.torch.batchify import batchify


class PytorchModule(GenericNetwork):
    """
    A wrapper class for PyTorch Modules.

    Parameters
    ----------
    network
        a PyTorch module
    """

    def __init__(
        self,
        network: torch.nn,
        device: Optional[torch.device] = None,
        dtype: Optional[DType] = np.float32,
        **kwargs
    ) -> None:
        super().__init__(network)
        self.device = device  # TODO implement; if device is not None else get_pytorch_context()
        self.dtype = dtype

    def forward_pass_numpy(
        self, inputs, dtype: Optional[DType] = np.float32
    ) -> np.ndarray:
        return self.network(inputs).data.numpy().astype(dtype)

    def get_forward_input_names(self):
        raise NotImplementedError
        # return get_forward_input_names(self.network)

    def get_batchify_fn(self, batchify_fn: Optional[Callable] = None):
        return partial(
            batchify if batchify_fn is None else batchify_fn,
            device=self.device,
        )

    def serialize_network(self, path: Path, name: str) -> None:
        raise NotImplementedError

    @classmethod
    def deserialize(cls, path: Path, name: str, **kwargs) -> "PytorchModule":
        raise NotImplementedError

    @classmethod
    def deserialize_network(
        cls,
        path: Path,
        name: str,
        ctx: Optional[mx.Context] = None,
        **parameters
    ) -> "PytorchModule":
        raise NotImplementedError
