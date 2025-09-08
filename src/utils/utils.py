import os
import sys
import time
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Dict, List

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only
import numpy as np
import torch
import nibabel as nib
import pandas as pd

import errno
import os
import signal
import functools

from src.utils import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)





def set_pythonhashseed(seed=0):
    current_seed = os.environ.get("PYTHONHASHSEED")
    seed = str(seed)
    if current_seed is None or current_seed != seed:
        print(f'Setting PYTHONHASHSEED="{seed}"')
        os.environ["PYTHONHASHSEED"] = seed
        # restart the current process
        os.execl(sys.executable, sys.executable, *sys.argv)


class TimeoutError(Exception):
    pass


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator


def save_outputs_as_nii(dict_list: List[dict]):
    for data_dict in dict_list:
        img_path = Path(data_dict['img_path'])
        out_dir = img_path.parents[1] / 'outputs'
        os.makedirs(out_dir, exist_ok=True)

        if os.path.exists(img_path):
            target = data_dict['targets'].squeeze().swapaxes(0, 1)
            output = data_dict['outputs'].squeeze().swapaxes(0, 1)
            error = data_dict['error'].squeeze().swapaxes(0, 1)

            if img_path.name.split('.')[1] == 'nii':
                img = nib.load(img_path)
                affine, header = img.affine, img.header
            else:
                affine, header = np.eye(4), nib.Nifti1Header()

            img_target = nib.Nifti1Image(target, affine=affine, header=header)
            img_output = nib.Nifti1Image(output, affine=affine, header=header)
            img_error = nib.Nifti1Image(error, affine=affine, header=header)

            nib.save(img_target, out_dir / f"{img_path.stem.split('.')[0]}_target.nii")
            nib.save(img_output, out_dir / f"{img_path.stem.split('.')[0]}_output.nii")
            nib.save(img_error, out_dir / f"{img_path.stem.split('.')[0]}_error.nii")

            if 'outputs_rss' in data_dict:
                output_rss = data_dict['outputs_rss'].squeeze().swapaxes(0, 1)
                img_output_rss = nib.Nifti1Image(output_rss, affine=affine, header=header)
                nib.save(img_output_rss, out_dir / f"{img_path.stem.split('.')[0]}_output_rss.nii")

            if 'outputs_cs' in data_dict:
                output_cs = data_dict['outputs_cs'].squeeze().swapaxes(0, 1)
                img_output_cs = nib.Nifti1Image(output_cs, affine=affine, header=header)
                nib.save(img_output_cs, out_dir / f"{img_path.stem.split('.')[0]}_output_cs.nii")

            if 'error_rss' in data_dict:
                error_rss = data_dict['error_rss'].squeeze().swapaxes(0, 1)
                img_error_rss = nib.Nifti1Image(error_rss, affine=affine, header=header)
                nib.save(img_error_rss, out_dir / f"{img_path.stem.split('.')[0]}_error_rss.nii")

            if 'error_cs' in data_dict:
                error_cs = data_dict['error_cs'].squeeze().swapaxes(0, 1)
                img_error_cs = nib.Nifti1Image(error_cs, affine=affine, header=header)
                nib.save(img_error_cs, out_dir / f"{img_path.stem.split('.')[0]}_error_cs.nii")

        else:
            print(f"Could not find file {img_path}")


def save_results_to_csv(dict_list: List[dict]):
    data = {key: [data_dict[key] for data_dict in dict_list] for key in list(dict_list[0].keys())[7:]}
    df = pd.DataFrame(data)

    img_path = Path(dict_list[0]['img_path'])
    out_dir = img_path.parents[1] / 'res'
    os.makedirs(out_dir, exist_ok=True)

    df.to_csv(f'{out_dir}/results.csv', index=False)


class TestOutputs(Callback):
    def __init__(self):
        self.test_outputs = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs):
        self.test_outputs.append(outputs)  # or extend

    def get_test_outputs(self):
        return self.test_outputs


def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)


def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
    """
    Converts a complex torch tensor to numpy array.

    Args:
        data: Input data to be converted to numpy.

    Returns:
        Complex numpy version of data.
    """
    return torch.view_as_complex(data).numpy()


def combine_selected_splits(splits, selected_indices):
    """
    Combines selected splits from a list of lists

    Args:
    - splits: List of lists
    - selected_indices: List of indices representing the selected splits to combine

    Returns:
    - Combined list from selected splits
    """
    combined_list = []
    for index in selected_indices:
        combined_list.extend(splits[index])

    return combined_list


def split_list_randomly(list, n_splits, seed=None):
    """
    Randomly splits a list into n_splits using a seed if provided.

    Args:
    - list: List to split
    - n_splits: Number of splits to create
    - seed: Seed for random splitting, if provided

    Returns:
    - List of lists
    """
    if seed is not None:
        prev_state = np.random.get_state()
        np.random.seed(seed)

    np.random.shuffle(list)

    if seed is not None:
        np.random.set_state(prev_state)

    k, m = divmod(len(list), n_splits)
    return [list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_splits)]


def shuffle_list(list, seed=None):
    """
    Randomly splits a list into n_splits using a seed if provided.

    Args:
    - list: List to split
    - n_splits: Number of splits to create
    - seed: Seed for random splitting, if provided

    Returns:
    - List of lists
    """
    if seed is not None:
        prev_state = np.random.get_state()
        np.random.seed(seed)

    np.random.shuffle(list)

    if seed is not None:
        np.random.set_state(prev_state)

    return list


def same_n_files(dirs):
    """
    Checks if multiple directories have the same number of files.

    Args:
    - dirs: List of directories to check

    Returns:
    - True if all directories have the same number of files, False otherwise
    """
    if not all(directory.exists() for directory in dirs): return False
    file_counts = [len(get_filepaths_of_dir(directory)) for directory in dirs]
    return all(count == file_counts[0] for count in file_counts)


def get_filepaths_of_dir(path: Path, ext: str = "*") -> List[Path]:
    """
    Returns a list of files in a directory/path. Uses pathlib.
    """
    filenames = [file for file in path.glob(ext) if file.is_file() and file.name != '.DS_Store']
    assert len(filenames) > 0, f"No files found in path: {path}"
    return filenames


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.

    Makes multirun more resistant to failure.

    Utilities:
    - Calling the `utils.extras()` before the task is started
    - Calling the `utils.close_loggers()` after the task is finished or failed
    - Logging the exception if occurs
    - Logging the output dir
    """

    def wrap(cfg: DictConfig):

        # execute the task
        try:

            # apply extra utilities
            extras(cfg)

            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:

            # save exception to `.log` file
            log.exception("")

            # when using hydra plugins like Optuna, you might want to disable raising exception
            # to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:

            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # close loggers (even if exception occurs so multirun won't fail)
            close_loggers()

        return metric_dict, object_dict

    return wrap


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config."""
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()


@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup)."""
    with open(path, "w+") as file:
        file.write(content)
