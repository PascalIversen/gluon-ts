# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# Standard library imports
import logging
import os
import tempfile
import time
import uuid
from typing import Any, List, Optional, Union

# Third-party imports

import numpy as np

# First-party imports
import torch
from gluonts.torch.utils import get_torch_device
from gluonts.core.component import validated
from gluonts.core.exception import GluonTSDataError, GluonTSUserError
from gluonts.dataset.loader import TrainDataLoader, ValidationDataLoader
from gluonts.gluonts_tqdm import tqdm

# Relative imports
from . import learning_rate_scheduler as lrs
from .model_averaging import (
    AveragingStrategy,
    SelectNBestMean,
    save_epoch_info,
)
from .model_iteration_averaging import (
    IterationAveragingStrategy,
    NTA,
    Alpha_Suffix,
)

logger = logging.getLogger("gluonts").getChild("trainer")


MODEL_ARTIFACT_FILE_NAME = "model"
STATE_ARTIFACT_FILE_NAME = "state"


def check_loss_finite(val: float) -> None:
    if not np.isfinite(val):
        raise GluonTSDataError(
            "Encountered invalid loss value! Try reducing the learning rate "
            "or try a different likelihood."
        )


# TODO do I need this
# def loss_value(loss: torch.metrics.Loss) -> float:
#     return loss.get_name_value()[0][1]


class Trainer:
    # FIXME PyTorchTrainer and GluonTrainer should inherit from common Trainer ABC
    # TODO implement

    @validated()
    def __init__(
        self,
        device: Optional[torch.device] = None,
        epochs: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        learning_rate: float = 1e-3,
        learning_rate_decay_factor: float = 0.5,
        patience: int = 10,
        minimum_learning_rate: float = 5e-5,
        clip_gradient: float = 10.0,
        weight_decay: float = 1e-8,
        # TODO implement user specified initalization: init: Union[str, mx.initializer.Initializer] = "xavier",
        avg_strategy: Union[
            AveragingStrategy, IterationAveragingStrategy
        ] = SelectNBestMean(num_models=1),
    ) -> None:

        assert (
            0 <= epochs < float("inf")
        ), "The value of `epochs` should be >= 0"
        assert 0 < batch_size, "The value of `batch_size` should be > 0"
        assert (
            0 < num_batches_per_epoch
        ), "The value of `num_batches_per_epoch` should be > 0"
        assert (
            0 < learning_rate < float("inf")
        ), "The value of `learning_rate` should be > 0"
        assert (
            0 <= learning_rate_decay_factor < 1
        ), "The value of `learning_rate_decay_factor` should be in the [0, 1) range"
        assert 0 <= patience, "The value of `patience` should be >= 0"
        assert (
            0 <= minimum_learning_rate
        ), "The value of `minimum_learning_rate` should be >= 0"
        assert 0 < clip_gradient, "The value of `clip_gradient` should be > 0"
        assert 0 <= weight_decay, "The value of `weight_decay` should be => 0"

        self.epochs = epochs
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.learning_rate = learning_rate
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.patience = patience
        self.minimum_learning_rate = minimum_learning_rate
        self.clip_gradient = clip_gradient
        self.weight_decay = weight_decay

        self.avg_strategy = avg_strategy
        self.device = device if device is not None else get_torch_device()
        self.halt = False

    def set_halt(self, signum: int, stack_frame: Any) -> None:
        logger.info("Received signal: {}".format(signum))
        self.halt = True

    def count_model_params(self, net: torch.nn) -> int:
        raise NotImplementedError

    def __call__(
        self,
        net: torch.nn,
        input_names: List[str],
        train_iter: TrainDataLoader,
        validation_iter: Optional[ValidationDataLoader] = None,
    ) -> None:  # TODO: we may want to return some training information here
        is_validation_available = validation_iter is not None
        self.halt = False

        with tempfile.TemporaryDirectory(
            prefix="gluonts-trainer-temp-"
        ) as gluonts_temp:

            def base_path() -> str:
                return os.path.join(
                    gluonts_temp,
                    "{}_{}".format(STATE_ARTIFACT_FILE_NAME, uuid.uuid4()),
                )

            logger.info("Start model training")

            batch_size = train_iter.batch_size

            best_epoch_info = {
                "params_path": "%s-%s.params" % (base_path(), "init"),
                "epoch_no": -1,
                "score": np.Inf,
            }

            optimizer = torch.optim.Adam(
                net.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

            # TODO implement PyTorch gradient clipping

            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=self.patience,
                factor=self.learning_rate_decay_factor,
                min_lr=self.minimum_learning_rate,
            )

            def loop(epoch_no, batch_iter, is_training: bool = True) -> float:
                tic = time.time()

                epoch_loss = 0.0

                # use averaged model for validation
                if not is_training and isinstance(
                    self.avg_strategy, IterationAveragingStrategy
                ):
                    self.avg_strategy.load_averaged_model(net)

                with tqdm(batch_iter) as it:
                    for batch_no, data_entry in enumerate(it, start=1):
                        if self.halt:
                            break

                        inputs = [data_entry[k] for k in input_names]

                        optimizer.zero_grad()

                        output = net(*inputs)

                        # network can returns several outputs, the first being always the loss
                        # when having multiple outputs, the forward returns a list in the case of hybrid and a
                        # tuple otherwise
                        # we may wrap network outputs in the future to avoid this type check
                        if isinstance(output, (list, tuple)):
                            loss = output[0]
                        else:
                            loss = output
                        # TODO # FIXME different reduction strategy?
                        loss = loss.mean()
                        if is_training:
                            loss.backward()
                            optimizer.step()
                            # iteration averaging in training
                            if isinstance(
                                self.avg_strategy, IterationAveragingStrategy,
                            ):
                                self.avg_strategy.apply(net)
                        epoch_loss += loss.detach().numpy()

                        if not np.isfinite(loss.detach().numpy()):
                            logger.warning("Epoch[%d] gave nan loss", epoch_no)
                            return loss.detach().numpy()

                        it.set_postfix(
                            ordered_dict={
                                "epoch": f"{epoch_no + 1}/{self.epochs}",
                                ("" if is_training else "validation_")
                                + "avg_epoch_loss": epoch_loss
                                / (batch_no + 1),
                            },
                            refresh=False,
                        )
                        # print out parameters of the network at the first pass # FIXME implement for torch
                        # if batch_no == 1 and epoch_no == 0:
                        #     net_name = type(net).__name__
                        #     num_model_param = self.count_model_params(net)
                        #     logger.info(
                        #         f"Number of parameters in {net_name}: {num_model_param}"
                        #     )
                # mark epoch end time and log time cost of current epoch
                toc = time.time()
                logger.info(
                    "Epoch[%d] Elapsed time %.3f seconds",
                    epoch_no,
                    (toc - tic),
                )

                logger.info(
                    "Epoch[%d] Evaluation metric '%s'=%f",
                    epoch_no,
                    ("" if is_training else "validation_") + "epoch_loss",
                    epoch_loss / (batch_no + 1),
                )

                if not is_training and isinstance(
                    self.avg_strategy, IterationAveragingStrategy
                ):
                    # bring back the cached model
                    self.avg_strategy.load_cached_model(net)

                return epoch_loss / (batch_no + 1)

            for epoch_no in range(self.epochs):
                if self.halt:
                    logger.info(f"Epoch[{epoch_no}] Interrupting training")
                    break

                curr_lr = optimizer.param_groups[0]["lr"]

                logger.info(f"Epoch[{epoch_no}] Learning rate is {curr_lr}")

                epoch_loss = loop(epoch_no, train_iter)
                if is_validation_available:
                    epoch_loss = loop(
                        epoch_no, validation_iter, is_training=False
                    )

                # update average trigger
                if isinstance(self.avg_strategy, IterationAveragingStrategy):
                    self.avg_strategy.update_average_trigger(
                        metric=epoch_loss, epoch=epoch_no + 1
                    )
                    # once triggered, update the average immediately
                    self.avg_strategy.apply(net)

                lr_scheduler.step(epoch_loss)
                if False:  # FIXME
                    logger.info("Stopping training")
                    break

                # save model and epoch info
                bp = base_path()
                epoch_info = {
                    "params_path": f"{bp}-0000.params",
                    "epoch_no": epoch_no,
                    "score": epoch_loss,
                }
                torch.save(
                    net.state_dict(), epoch_info["params_path"]
                )  # TODO: handle possible exception

                save_epoch_info(bp, epoch_info)

                # update best epoch info - needed for the learning rate scheduler
                if epoch_loss < best_epoch_info["score"]:
                    best_epoch_info = epoch_info.copy()

                if (
                    not optimizer.param_groups[0]["lr"] == curr_lr
                ):  # fixme: is this correct? what is this supposed to do?
                    if best_epoch_info["epoch_no"] == -1:
                        raise GluonTSUserError(
                            "Got NaN in first epoch. Try reducing initial learning rate."
                        )

                    logger.info(
                        f"Loading parameters from best epoch "
                        f"({best_epoch_info['epoch_no']})"
                    )
                    net.load_state_dict(
                        torch.load(best_epoch_info["params_path"])
                    )
                    net.to(self.device)

            if isinstance(self.avg_strategy, AveragingStrategy):
                logging.info("Computing averaged parameters.")
                averaged_params_path = self.avg_strategy.apply(gluonts_temp)

                logging.info("Loading averaged parameters.")
                net.load_state_dict(torch.load(averaged_params_path))
                net.to(self.device)

            if isinstance(self.avg_strategy, IterationAveragingStrategy):
                logging.info("Loading averaged parameters.")
                self.avg_strategy.load_averaged_model(net)

            logger.info("End model training")
