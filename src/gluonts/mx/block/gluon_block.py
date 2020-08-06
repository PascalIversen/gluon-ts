from functools import partial

from gluonts.core.component import get_mxnet_context, DType
from gluonts.core.serde import dump_json, load_json
from gluonts.generic_network import GenericNetwork
import mxnet as mx
from pathlib import Path
import numpy as np
from gluonts.support.util import (
    get_hybrid_forward_input_names,
    export_repr_block,
    import_symb_block,
)
from gluonts.dataset.loader import DataBatch
from gluonts.support.util import export_symb_block, import_repr_block
from typing import Optional, Tuple, Dict, Callable

from gluonts.mx.batchify import batchify


class GluonBlock(GenericNetwork):
    """
    A wrapper class for Gluon Blocks.

    Parameters
    ----------
    network
        a Gluon Block
    """

    batchify_fn = batchify

    def __init__(
        self,
        network: mx.gluon.Block,
        ctx: Optional[mx.Context] = None,
        dtype: Optional[DType] = np.float32,
        **kwargs
    ) -> None:
        super().__init__(network)
        self.ctx = ctx
        self.dtype = dtype

    def forward_pass_numpy(
        self, inputs, dtype: Optional[DType] = np.float32
    ) -> np.ndarray:
        return self.network(inputs).asnumpy(dtype=dtype)

    def get_forward_input_names(self):
        return get_hybrid_forward_input_names(self.network)

    def get_batchify_fn(self, batchify_fn: Optional[Callable]):
        return (
            partial(
                self.batchify_fn if batchify_fn is None else batchify_fn,
                ctx=self.ctx,
                dtype=self.dtype,
            ),
        )

    def serialize_network(self, path: Path, name: str) -> None:
        with (path / "contextual_parameters.json").open("w") as fp:
            contextual_parameters = dict(ctx=self.ctx, dtype=self.dtype)
            print(dump_json(contextual_parameters), file=fp)

    @classmethod
    def deserialize_block(
        cls, path: Path, name: str, **kwargs
    ) -> "GluonBlock":
        raise NotImplementedError

    @classmethod
    def deserialize_network(
        cls,
        path: Path,
        name: str,
        ctx: Optional[mx.Context] = None,
        **parameters
    ) -> "GluonBlock":
        ctx = ctx if ctx is not None else get_mxnet_context()

        with (path / "contextual_parameters.json").open("r") as fp:
            contextual_parameters = load_json(fp.read())

        contextual_parameters["ctx"] = ctx

        return cls.deserialize_block(
            path, name, **contextual_parameters, **parameters
        )


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
        super().serialize_network(path, name)
        export_repr_block(self.network, path, name)

    # pylint:disable=arguments-differ
    @classmethod
    def deserialize_block(
        cls, path: Path, name: str, ctx: mx.Context, **kwargs
    ) -> "GluonHybridBlock":
        with mx.Context(ctx):
            # deserialize network
            network = import_repr_block(path, name)

        return GluonHybridBlock(network=network, ctx=ctx, **kwargs)

    def get_forward_input_names(self, network: "GluonHybridBlock"):
        return get_hybrid_forward_input_names(network.network)


class GluonSymbolBlock(GluonBlock):

    """
    A wrapper class for Gluon SymbolBlocks.

    Parameters
    ----------
    network
        a Gluon SymbolBlock
    """

    def __init__(self, network: mx.gluon.SymbolBlock, **kwargs) -> None:
        super().__init__(network, **kwargs)

    def serialize_network(self, path: Path, name: str) -> None:
        super().serialize_network(path, name)
        export_symb_block(self.network, path, name)

    # pylint:disable=arguments-differ
    @classmethod
    def deserialize_block(
        cls,
        path: Path,
        name: str,
        ctx: mx.Context,
        input_names: Dict,
        **kwargs
    ) -> "GluonSymbolBlock":
        with mx.Context(kwargs["ctx"]):
            # deserialize network
            network = import_symb_block(len(input_names), path, name)

        # for some reason we don't want to pass input names
        return GluonSymbolBlock(network=network, ctx=ctx, **kwargs)
