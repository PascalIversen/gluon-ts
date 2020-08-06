from gluonts.core.component import get_mxnet_context
from gluonts.generic_network import GenericNetwork
import mxnet as mx
from pathlib import Path
import numpy as np
from gluonts.support.util import get_hybrid_forward_input_names
from gluonts.dataset.loader import DataBatch
from gluonts.support.util import export_symb_block, import_repr_block
from typing import Optional, Tuple, Dict


class GluonBlock(GenericNetwork):
    """
    A wrapper class for Gluon Blocks.

    Parameters
    ----------
    network
        a Gluon Block
    """

    def __init__(self, network: mx.gluon.Block) -> None:
        super().__init__(network)

    def forward_pass_numpy(self, inputs) -> np.ndarray:
        return self.network(inputs).asnumpy()

    def get_forward_input_names(self):
        return get_hybrid_forward_input_names(self.network)


class GluonHybridBlock(GluonBlock):
    """
    A wrapper class for Gluon HybridBlocks.

    Parameters
    ----------
    network
        a Gluon HybridBlock
    """

    # convenience method: not necessary
    def hybridize(self, batch: DataBatch) -> None:
        """
        Hybridizes the underlying prediction network.

        Parameters
        ----------
        batch
            A batch of data to use for the required forward pass after the
            `hybridize()` call.
        """
        self.network.hybridize(active=True)
        self.network(*[batch[k] for k in self.get_forward_input_names()])

    def serialize_network(self, path: Path, name: str) -> None:
        export_symb_block(self.network, path, name)

    @classmethod
    def deserialize_network(
        cls, path: Path, name: str, ctx: Optional[mx.Context] = None, **kwargs
    ) -> Tuple["GluonHybridBlock", Dict]:
        ctx = ctx if ctx is not None else get_mxnet_context()

        with mx.Context(ctx):
            # deserialize network
            network = import_repr_block(path, name)
            contextual_parameters = {"ctx": ctx}

        return GluonHybridBlock(network=network), contextual_parameters
