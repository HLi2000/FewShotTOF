from typing import List
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
import time
import h5py
import torch
import pathlib
import numpy as np

from src.data.components.transforms import normalize_01
from src.utils.coil import CoilCompressorSVD
from src.utils.map import undersample_mask, EspiritCalib
from src.utils.mri_utils import simulate_maps, fftnc, simulate_phase, ifftnc, unnormalise_target, read_into_slabs_overlap
from src.utils.utils import to_tensor, timeout, TimeoutError
from src.data.components.subsample import create_mask_for_mask_type


class IXIDataSet(torch.utils.data.Dataset):
    """
    Builds a dataset with images
    """

    def __init__(self,
                 inputs: List[pathlib.Path],
                 transform = None,
                 transform_kspace = None,
                 transform_target = None,
                 acc_rate = 1.0,
                 slope = 10,
                 calib = (12,6),
                 num_compressed_coils = None,
                 partial_fourier = 0.,
                 test: bool = False,
                 estimated_maps: bool = False,
                 mask: str = 'poisson2d',
                 augment: bool = False,
                 ):
        self.inputs = inputs
        self.transform = transform
        self.transform_kspace = transform_kspace
        self.transform_target = transform_target
        self.acc_rate = acc_rate
        self.slope = slope
        self.calib = calib
        self.num_compressed_coils = num_compressed_coils
        self.partial_fourier = partial_fourier
        self.seed = np.random.get_state()[1][0]
        self.test = test
        self.estimated_maps = estimated_maps
        self.mask = mask
        self.current_epoch = 0
        self.augment = augment

    def __len__(self):
        return len(self.inputs)

    def augment_data(self, target: torch.Tensor, kspace: torch.Tensor, seed=None):
        """
        Augments the given data and kspace with random flipping and 90-degree rotations.
        Expects data with dimensions [channel, x, y, z].
        """
        if seed is not None:
            # Save the current random state
            rng_state = torch.random.get_rng_state()
            torch.manual_seed(seed)

        if torch.rand(1).item() > 0.5:
            target = torch.flip(target, dims=[2]).clone()  # Horizontal flip along y-axis
            target = torch.roll(target, shifts=1, dims=2)
            kspace = torch.flip(kspace, dims=[2]).clone()

        if seed is not None:
            # Restore the random state
            torch.random.set_rng_state(rng_state)

        return target, kspace

    def __getitem__(self,
                    index: int):
        img_path = self.inputs[index]

        # paths
        filename = img_path.stem.split('.')[0]

        if self.test:
            self.seed = int(str(hash(filename))[-8:])

        # Load inputs
        if img_path.suffix in ('.npy', '.npz'):
            slabs = np.load(img_path)
        else:
            slabs = read_into_slabs_overlap(img_path, n_slabs=5, slab_size=24, overlap=0.2)
        target = abs(slabs) # [x, y, z, sl]

        # Un-normalisation for TOF
        target = unnormalise_target(target, seed=int(str(hash(filename))[-8:]))

        # Set seed
        torch.manual_seed(int(str(hash(filename))[-8:]))

        h5_path = img_path.parents[1] / 'preprocessed24' / f"{filename}.h5"
        if h5_path.is_file():
            with h5py.File(h5_path, 'r') as hf:
                compressed_kspaces = torch.from_numpy(hf['compressed_kspaces'][:])
                if self.estimated_maps:
                    estimated_maps = torch.from_numpy(hf['estimated_maps'][:])

            if self.transform is not None:
                target = normalize_01(self.transform(target), eps=torch.finfo(torch.float32).eps) + torch.finfo(torch.float32).eps

            phs = torch.ones_like(target)

            # undersample mask for one slab [coil, x, y, z, 2]
            print(f'Generating undersampling mask using seed={self.seed}')
            while True:
                try:
                    @timeout(5)  # Timeout set to n seconds
                    def generate_mask():
                        if self.mask == 'poisson2d':
                            return undersample_mask(target.shape[:3], r=self.acc_rate, slope=self.slope,
                                                type='poiss', n_dim=3, calib=self.calib, seed=self.seed), self.calib
                        elif self.mask == 'poisson1d':
                            return undersample_mask(target.shape[:3], r=self.acc_rate, slope=self.slope,
                                                    type='poiss', n_dim=1, calib=self.calib, seed=self.seed), self.calib[0]
                        else:
                            n_x, n_y, n_z = target.shape[:3]
                            if self.mask == 'equispaced':
                                mask_func = create_mask_for_mask_type("equispaced", [self.calib[0]/n_y], [int(self.acc_rate)])
                            elif self.mask == 'random1d':
                                mask_func = create_mask_for_mask_type("random", [self.calib[0]/n_y], [int(self.acc_rate)])
                            mask, num_acs = mask_func([1, n_y, 1], seed=self.seed)
                            mask = np.stack([np.stack([mask.squeeze().numpy()] * n_z, axis=-1)] * n_x, axis=0).astype(bool)
                            return mask, num_acs
                    mask, num_acs = generate_mask()

                    print('Undersampling mask is generated')
                    break
                except TimeoutError:
                    self.seed += 1
                    print(f'Mask generation timed out. Retrying with seed={self.seed}')
                    continue
            mask = torch.stack([torch.stack([to_tensor(mask)] * compressed_kspaces.shape[0], dim=0)] * 2, dim=-1)
            if self.partial_fourier > 0:
                mask[:, :int(mask.shape[1] * self.partial_fourier)] = 0

            # Apply augmentation to each slab of the target and k-space
            if self.augment and not self.test:
                augmented_targets = []
                augmented_kspaces = []
                for slab in range(target.shape[-1]):
                    augmented_target, augmented_kspace = self.augment_data(target[...,slab].unsqueeze(0), compressed_kspaces[...,slab],
                                                            seed=(self.seed + slab))
                    augmented_targets.append(augmented_target.squeeze(0))
                    augmented_kspaces.append(augmented_kspace)
                target = torch.stack(augmented_targets, dim=-1)
                compressed_kspaces = torch.stack(augmented_kspaces, dim=-1)

            masked_kspaces = [compressed_kspaces[...,slb] * mask[...,0] + 0. for slb in range(compressed_kspaces.shape[4])]
            masked_kspaces = torch.stack(masked_kspaces, dim=-1)

            data = {'masked_kspaces': masked_kspaces,
                    'estimated_maps': estimated_maps if self.estimated_maps else 0.,
                    'mask': mask,
                    'targets': target * torch.exp(1j * phs),
                    'data_range': target.max()-target.min(),
                    'img_path': str(img_path)
                    }

            if self.test:
                self.val_test_data[filename] = data
            else:
                self.seed += 1

            return data

        if self.transform is not None:
            target = normalize_01(self.transform(target), eps=torch.finfo(torch.float32).eps) + torch.finfo(torch.float32).eps

        # Generate coil maps on-the-fly [c, x, y, z, sl]
        maps, skull_masks = simulate_maps(slb_data=target.numpy(), seed=int(str(hash(filename))[-8:]))
        maps = torch.from_numpy(maps)

        # Generate phase maps on-the-fly [x, y, z, sl]
        phs = torch.from_numpy(simulate_phase(slb_data=target.numpy(), seed=int(str(hash(filename))[-8:])))

        # combine to simulate a multi-coil multi-slab complex input
        input = target * torch.exp(1j * phs)
        input = torch.stack([input] * maps.shape[0], dim=0).type(torch.complex64) * maps

        # undersample mask for one slab [coil, x, y, z, 2]
        print(f'Generating undersampling mask using seed={self.seed}')
        while True:
            try:
                @timeout(5)  # Timeout set to n seconds
                def generate_mask():
                    if self.mask == 'poisson2d':
                        return undersample_mask(target.shape[:3], r=self.acc_rate, slope=self.slope,
                                            type='poiss', n_dim=3, calib=self.calib, seed=self.seed), self.calib
                    elif self.mask == 'poisson1d':
                        return undersample_mask(target.shape[:3], r=self.acc_rate, slope=self.slope,
                                                type='poiss', n_dim=1, calib=self.calib, seed=self.seed), self.calib[0]
                    else:
                        n_x, n_y, n_z = target.shape[:3]
                        if self.mask == 'equispaced':
                            mask_func = create_mask_for_mask_type("equispaced", [self.calib[0]/n_y], [int(self.acc_rate)])
                        elif self.mask == 'random1d':
                            mask_func = create_mask_for_mask_type("random", [self.calib[0]/n_y], [int(self.acc_rate)])
                        mask, num_acs = mask_func([1, n_y, 1], seed=self.seed)
                        mask = np.stack([np.stack([mask.squeeze().numpy()] * n_z, axis=-1)] * n_x, axis=0).astype(bool)
                        return mask, num_acs
                mask, num_acs = generate_mask()

                print('Undersampling mask is generated')
                break
            except TimeoutError:
                self.seed += 1
                print(f'Mask generation timed out. Retrying with seed={self.seed}')
                continue

        mask = torch.stack([torch.stack([to_tensor(mask)] * maps.shape[0], dim=0)] * 2, dim=-1)
        if self.partial_fourier > 0:
            mask[:, :int(mask.shape[1] * self.partial_fourier)] = 0

        # Compression first
        compressed_kspaces = []
        for slb in range(input.shape[4]):
            slab = input[:, :, :, :, slb]
            compressed_kspace = fftnc(slab, dim=[1, 2, 3]) + 0.

            # Coil compression
            if self.num_compressed_coils is not None:
                compressor = CoilCompressorSVD(out_coils=self.num_compressed_coils, is_3D=False)
                compressed_kspace = torch.from_numpy(compressor.compress(compressed_kspace.numpy()))

            compressed_kspaces.append(compressed_kspace)
        compressed_kspaces = np.stack(compressed_kspaces, axis=-1)
        compressed_kspaces = torch.from_numpy(compressed_kspaces)
        mask = mask[:self.num_compressed_coils, ...] if self.num_compressed_coils is not None else mask
        masked_kspaces = [compressed_kspaces[..., slb] * mask[..., 0] + 0. for slb in range(compressed_kspaces.shape[4])]
        masked_kspaces = torch.stack(masked_kspaces, dim=-1)

        # Estimate maps using ESPIRiT
        if self.estimated_maps:
            t = time.time()
            estimated_maps = torch.zeros_like(masked_kspaces)
            for slb in range(masked_kspaces.shape[4]):
                masked_kspace = masked_kspaces[:, :, :, :, slb]
                masked_kspace = ifftnc(masked_kspace, dim=[1])

                for x in range(masked_kspace.shape[1]):
                    estimated_maps[:, x, :, :, slb] = torch.from_numpy(EspiritCalib(np.asarray(masked_kspace[:, x, ...]),
                                        calib_sz=self.calib, kernel_width=3, thresh=0.02, crop=0.9, show_pbar=False).run())
            estimated_maps = torch.stack([estimated_maps.real, estimated_maps.imag], dim=-1)
            print('ESPIRiT elapsed: {:.2f}s'.format(time.time() - t))


        # Save data to h5 file
        os.makedirs(h5_path.parent, exist_ok=True)
        with h5py.File(h5_path, 'w') as hf:
            hf.create_dataset('compressed_kspaces', data=compressed_kspaces.numpy())
            if self.estimated_maps:
                hf.create_dataset('estimated_maps', data=estimated_maps.numpy())

        data = {'masked_kspaces': masked_kspaces,
                'estimated_maps': estimated_maps if self.estimated_maps else 0.,
                'mask': mask,
                'targets': target * torch.exp(1j * phs),
                'data_range': target.max()-target.min(),
                'img_path': str(img_path)
                }

        if not self.test:
            self.seed += 1

        return data