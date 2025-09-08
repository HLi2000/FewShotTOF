from typing import List, Tuple

import hydra
import pyrootutils
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src import utils
from src.utils.utils import TestOutputs, set_pythonhashseed

log = utils.get_pylogger(__name__)

# # Test BART's setup
# from bart import bart
# import numpy as np
# mask = bart(1, 'pattern', np.random.rand(10,10))


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)
        set_pythonhashseed(cfg.seed)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    test_outputs = TestOutputs()
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=[test_outputs])

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting testing!")
    # trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    # # pretrained only
    # # model.net.load_state_dict(torch.load(cfg.get("ckpt_path"), map_location=torch.device('cpu')))
    # model.net.load_state_dict(torch.load(cfg.get("ckpt_path")))
    # trainer.test(model=model, datamodule=datamodule)
    # # Parameters only
    if cfg.get("ckpt_path"):
        model.load_state_dict(torch.load(cfg.get("ckpt_path"), map_location=torch.device('cpu'))["state_dict"], strict=False)
        # model.load_state_dict(torch.load(cfg.get("ckpt_path"))["state_dict"], strict=False)
        trainer.test(model=model, datamodule=datamodule)
    else:
        trainer.test(model=model, datamodule=datamodule)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    # get test outputs
    outputs = test_outputs.get_test_outputs()

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
