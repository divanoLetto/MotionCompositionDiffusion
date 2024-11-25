"""
CSV logger
----------

CSV logger for basic experiment logging that does not require opening ports

"""
import logging
import os
from argparse import Namespace
from typing import Any, Dict, Optional, Union

# from lightning_fabric.loggers.csv_logs import _ExperimentWriter as _FabricExperimentWriter
# from lightning_fabric.loggers.csv_logs import CSVLogger as FabricCSVLogger
# Local replacement
from .csv_fabric import _ExperimentWriter as _FabricExperimentWriter
from .csv_fabric import CSVLogger as FabricCSVLogger

from lightning_fabric.loggers.logger import rank_zero_experiment
from lightning_fabric.utilities.logger import _convert_params
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.core.saving import save_hparams_to_yaml
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

log = logging.getLogger(__name__)


class ExperimentWriter(_FabricExperimentWriter):
    r"""Experiment writer for CSVLogger.

    Currently, supports to log hyperparameters and metrics in YAML and CSV
    format, respectively.

    Args:
        log_dir: Directory for the experiment logs
    """

    NAME_HPARAMS_FILE = "hparams.yaml"

    def __init__(self, log_dir: str) -> None:
        super().__init__(log_dir=log_dir)
        self.hparams: Dict[str, Any] = {}

    def log_hparams(self, params: Dict[str, Any]) -> None:
        """Record hparams."""
        self.hparams.update(params)

    def save(self) -> None:
        """Save recorded hparams and metrics into files."""
        hparams_file = os.path.join(self.log_dir, self.NAME_HPARAMS_FILE)
        save_hparams_to_yaml(hparams_file, self.hparams)
        return super().save()


class CSVLogger(Logger, FabricCSVLogger):
    r"""Log to local file system in yaml and CSV format.

    Logs are saved to ``os.path.join(save_dir, name)``.

    Example:
        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.loggers import CSVLogger
        >>> logger = CSVLogger("logs", name="my_exp_name")
        >>> trainer = Trainer(logger=logger)

    Args:
        save_dir: Save directory
        name: Experiment name. Defaults to ``'lightning_logs'``.
        prefix: A string to put at the beginning of metric keys.
        flush_logs_every_n_steps: How often to flush logs to disk (defaults to every 100 steps).
    """

    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        save_dir: _PATH,
        name: str = "lightning_logs",
        prefix: str = "",
        flush_logs_every_n_steps: int = 100,
    ):
        super().__init__(
            root_dir=save_dir,
            name=name,
            prefix=prefix,
            flush_logs_every_n_steps=flush_logs_every_n_steps,
        )
        self._save_dir = os.fspath(save_dir)

    @property
    def root_dir(self) -> str:
        """Parent directory for all checkpoint subdirectories.

        If the experiment name parameter is an empty string, no experiment subdirectory is used and the checkpoint will
        be saved in "save_dir/"
        """
        return os.path.join(self.save_dir, self.name)

    @property
    def log_dir(self) -> str:
        """The log directory for this run."""
        return self.root_dir

    @property
    def save_dir(self) -> str:
        """The current directory where logs are saved.

        Returns:
            The path to current directory where logs are saved.
        """
        return self._save_dir

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        # don't log hyperparameters
        # already done in the config
        return

    @property
    @rank_zero_experiment
    def experiment(self) -> _FabricExperimentWriter:
        r"""

        Actual _ExperimentWriter object. To use _ExperimentWriter features in your
        :class:`~pytorch_lightning.core.module.LightningModule` do the following.

        Example::

            self.logger.experiment.some_experiment_writer_function()

        """
        if self._experiment is not None:
            return self._experiment

        self._fs.makedirs(self.root_dir, exist_ok=True)
        self._experiment = ExperimentWriter(log_dir=self.log_dir)
        return self._experiment
