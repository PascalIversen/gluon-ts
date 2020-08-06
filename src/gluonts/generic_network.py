from pathlib import Path
import numpy as np
from typing import Any, Callable, Optional

from gluonts.core.component import DType


class GenericNetwork:
    """
    Abstract wrapper class for neural network objects.

    Parameters
    ----------
    network
        The framework-dependent neural network.
    """

    batchify_fn: Callable = None

    def __init__(self, network: Any, **kwargs) -> None:
        self.network = network

    def forward_pass(self, inputs) -> Any:
        """
        Forward computation on the underlying network.

        Parameters
        ----------
        inputs
            Input tensor for which to calculate the forward computations.

        Returns
        -------
        Tensor
            framework specific tensor
        """
        return self.network(inputs)

    def forward_pass_numpy(
        self, inputs, dtype: Optional[DType] = np.float32
    ) -> np.ndarray:
        """
        Forward computation on the underlying network.

        Parameters
        ----------
        inputs
            Input tensor for which to calculate the forward computations.

        Returns
        -------
        Tensor
            numpy array
        """
        raise NotImplementedError

    def serialize_network(self, path: Path, name: str) -> None:
        """
        Serialize the network structure
        """
        raise NotImplementedError

    @classmethod
    def deserialize_network(cls, path: Path, name: str, **kwargs) -> None:
        """
        Deserialize a network structure
        """
        raise NotImplementedError

    def get_forward_input_names(self):
        """

        """
        raise NotImplementedError

    def get_batchify_fn(self, batchify_fn: Optional[Callable]):
        """

        """
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError
