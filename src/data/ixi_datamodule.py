from typing import Any, Dict, Optional, Tuple
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
import torchio as tio
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms, InterpolationMode
from pathlib import Path
from src.utils.utils import get_filepaths_of_dir, split_list_randomly, combine_selected_splits
from src.data.components.transforms import ComposeSingle, FunctionWrapperSingle
from src.data.datasets.ixi_dataset import IXIDataSet
from src.data.datasets.raw_tof_dataset import RawTOFDataSet

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchio.data.image")



class IXIDataModule(LightningDataModule):
    """LightningDataModule for IXI dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        acc_rate=2.0,
        slope=10,
        calib=[12, 6],
        resize=None,
        num_compressed_coils=None,
        partial_fourier=0.,
        test_raw_dir=None,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        estimated_maps: bool = False,
        mask: str = 'poisson2d',
        augment: bool = False,
        prospective: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.hparams.calib = tuple(calib)

        # paths
        self.data_dir = Path(self.hparams.data_dir)
        self.img_dir = self.data_dir / 'images'

        # TorchIO
        transforms_tio = tio.Compose([
            tio.RandomBiasField(coefficients=0.3, order=3, p=1.0),
        ])

        # data transformations
        self.transforms = ComposeSingle([
            FunctionWrapperSingle(torch.from_numpy),

            FunctionWrapperSingle(torch.movedim, source=[0, 1], destination=[2, 3]) if not resize == None \
                else FunctionWrapperSingle(lambda x : x),
            transforms.Resize(tuple(resize), InterpolationMode.NEAREST) if not resize == None else FunctionWrapperSingle(lambda x : x),
            FunctionWrapperSingle(torch.movedim, source=[2, 3], destination=[0, 1]) if not resize == None \
                else FunctionWrapperSingle(lambda x : x),

            FunctionWrapperSingle(torch.movedim, source=[3], destination=[0]),
            FunctionWrapperSingle(lambda x: transforms_tio(tio.ScalarImage(tensor=x)).data),
            FunctionWrapperSingle(torch.movedim, source=[0], destination=[3]),
        ])

        # data transformations
        self.transforms_target = ComposeSingle([
            FunctionWrapperSingle(torch.from_numpy),
        ])

        # kspace transformations
        self.transforms_kspace = ComposeSingle([
            FunctionWrapperSingle(torch.from_numpy),
        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            img_paths = sorted(get_filepaths_of_dir(self.img_dir))
            splits = split_list_randomly(img_paths, 5, seed=1234)

            img_paths_train = combine_selected_splits(splits, [2, 3, 4])
            img_paths_val = splits[1]
            if self.hparams.test_raw_dir is None:
                img_paths_test = splits[0]
            else:
                img_paths_raw = sorted(get_filepaths_of_dir(Path(self.hparams.test_raw_dir)))
                img_paths_test = [img_paths_raw[i] for i in [0, 2, 9, 10, 12]] # Test - Retro
                # img_paths_test = [img_paths_raw[i] for i in [1, 7, 8, 14, 17]]  # Prospective
                # img_paths_train = [img_paths_raw[i] for i in [4]] # Fine-tuning
                # img_paths_val = [img_paths_raw[i] for i in [11]] # Validation

            self.data_train = IXIDataSet(
                inputs=img_paths_train,
                transform=self.transforms,
                acc_rate = self.hparams.acc_rate,
                slope = self.hparams.slope,
                calib = self.hparams.calib,
                num_compressed_coils=self.hparams.num_compressed_coils,
                partial_fourier=self.hparams.partial_fourier,
                estimated_maps=self.hparams.estimated_maps,
                mask=self.hparams.mask,
                augment=self.hparams.augment
            )
            # self.data_train = RawTOFDataSet(
            #     inputs=img_paths_train,
            #     transform_target=self.transforms_target,
            #     transform_kspace=self.transforms_kspace,
            #     acc_rate=self.hparams.acc_rate,
            #     slope=self.hparams.slope,
            #     calib=self.hparams.calib,
            #     num_compressed_coils=self.hparams.num_compressed_coils,
            #     estimated_maps=self.hparams.estimated_maps,
            #     mask=self.hparams.mask,
            #     prospective=self.hparams.prospective,
            # )
            self.data_val = IXIDataSet(
                inputs=img_paths_val,
                transform=self.transforms,
                acc_rate=self.hparams.acc_rate,
                slope=self.hparams.slope,
                calib=self.hparams.calib,
                num_compressed_coils=self.hparams.num_compressed_coils,
                partial_fourier=self.hparams.partial_fourier,
                test=True,
                estimated_maps=self.hparams.estimated_maps,
                mask=self.hparams.mask,
            )
            # self.data_val = RawTOFDataSet(
            #     inputs=img_paths_val,
            #     transform_target=self.transforms_target,
            #     transform_kspace=self.transforms_kspace,
            #     acc_rate=self.hparams.acc_rate,
            #     slope=self.hparams.slope,
            #     calib=self.hparams.calib,
            #     num_compressed_coils=self.hparams.num_compressed_coils,
            #     test=True,
            #     estimated_maps=self.hparams.estimated_maps,
            #     mask=self.hparams.mask,
            #     prospective=self.hparams.prospective,
            # )
            if self.hparams.test_raw_dir is None:
                self.data_test = IXIDataSet(
                    inputs=img_paths_test,
                    transform=self.transforms,
                    acc_rate=self.hparams.acc_rate,
                    slope=self.hparams.slope,
                    calib=self.hparams.calib,
                    num_compressed_coils=self.hparams.num_compressed_coils,
                    partial_fourier=self.hparams.partial_fourier,
                    test=True,
                    estimated_maps=self.hparams.estimated_maps,
                    mask=self.hparams.mask,
                )
            else:
                self.data_test = RawTOFDataSet(
                    inputs=img_paths_test,
                    transform_target=self.transforms_target,
                    transform_kspace=self.transforms_kspace,
                    acc_rate=self.hparams.acc_rate,
                    slope=self.hparams.slope,
                    calib=self.hparams.calib,
                    num_compressed_coils=self.hparams.num_compressed_coils,
                    test=True,
                    estimated_maps=self.hparams.estimated_maps,
                    mask=self.hparams.mask,
                    prospective=self.hparams.prospective,
                )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = IXIDataModule()
