from typing import List

import os
import h5py
import torch
import pathlib
import numpy as np

from src.data.components.subsample import create_mask_for_mask_type
from src.utils.coil import CoilCompressorSVD
from src.utils.map import undersample_mask, EspiritCalib
from src.utils.mri_utils import ifftnc
from src.utils.utils import to_tensor, timeout, TimeoutError



class RawTOFDataSet(torch.utils.data.Dataset):
    """
    Builds a dataset with images
    """

    def __init__(self,
                 inputs: List[pathlib.Path],
                 transform_target = None,
                 transform_kspace = None,
                 acc_rate = 1.0,
                 slope = 10,
                 calib = (12,6),
                 num_compressed_coils=None,
                 test=False,
                 estimated_maps: bool = False,
                 mask: str = 'poisson2d',
                 prospective: bool = False,
                 ):
        self.inputs = inputs
        self.transform_target = transform_target
        self.transform_kspace = transform_kspace
        self.acc_rate = acc_rate
        self.slope = slope
        self.calib = calib
        self.num_compressed_coils = num_compressed_coils
        self.seed = np.random.get_state()[1][0]
        self.test = test
        self.estimated_maps = estimated_maps
        self.mask = mask
        self.prospective = prospective

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        # Select the sample
        img_path = self.inputs[index]

        # paths
        filename = '_'.join(img_path.stem.split('_')[:-1])
        raw_path = img_path.parents[1] / 'raw' / f"{filename}_centre.npy"

        self.seed = int(str(hash(filename))[-8:])

        # Load inputs
        target = np.load(img_path)  # [x, y, z, sl] complex
        kspace = np.load(raw_path) # [c, x, y, z, sl]

        h5_path = img_path.parents[1] / 'preprocessed24_8r' / f"{filename}.h5"
        if h5_path.is_file():
            with h5py.File(h5_path, 'r') as hf:
                masked_kspaces = torch.from_numpy(hf['masked_kspaces'][:])
                if self.estimated_maps:
                    estimated_maps = torch.from_numpy(hf['estimated_maps'][:])

            if self.transform_target is not None:
                target = self.transform_target(abs(target)) * torch.exp(1j * self.transform_target(np.angle(target)))

            # undersample mask for one slab [coil, x, y, z, 2]
            if self.prospective is not True:
                print(f'Generating undersampling mask using seed={self.seed}')
                while True:
                    try:
                        @timeout(5)  # Timeout set to n seconds
                        def generate_mask():
                            if self.mask == 'poisson2d':
                                return undersample_mask(masked_kspaces.shape[1:4], r=self.acc_rate, slope=self.slope,
                                                    type='poiss', n_dim=3, calib=self.calib, seed=self.seed), self.calib
                            elif self.mask == 'poisson1d':
                                return undersample_mask(masked_kspaces.shape[1:4], r=self.acc_rate, slope=self.slope,
                                                        type='poiss', n_dim=1, calib=self.calib, seed=self.seed), self.calib[0]
                            else:
                                n_x, n_y, n_z = masked_kspaces.shape[1:4]
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

                mask = torch.stack([torch.stack([to_tensor(mask)] * kspace.shape[0], dim=0)] * 2, dim=-1)
                mask = mask[:self.num_compressed_coils, ...] if self.num_compressed_coils is not None else mask
            else:
                mask = torch.stack([abs(masked_kspaces[..., 0]) > 0] * 2, dim=-1)

            # Debug check: compare the actual data in the masks
            mask_match = torch.all((abs(masked_kspaces[0, -1, :, :, 0]) > 0) == (mask[0, -1, :, :, 0] > 0))
            print(f"[DEBUG] Do the masks match? {mask_match.item()}")

            return {'masked_kspaces': masked_kspaces,
                    'estimated_maps': estimated_maps if self.estimated_maps else 0.,
                    'mask': mask,
                    'targets': target,
                    'data_range': abs(target).max() - abs(target).min(),
                    'img_path': str(img_path)
                    }


        if self.transform_target is not None:
            target = self.transform_target(abs(target)) * torch.exp(1j * self.transform_target(np.angle(target)))

        if self.transform_kspace is not None:
            kspace_transformed_c = []
            for c in range(kspace.shape[0]):
                kspace_transformed_sl = []
                for sl in range(kspace.shape[-1]):
                    kspace_transformed_sl.append(self.transform_kspace(kspace[c, :, :, :, sl]))
                kspace_transformed_c.append(torch.stack(kspace_transformed_sl, dim=-1))
            kspace = torch.stack(kspace_transformed_c, dim=0)

        # undersample mask for one slab [coil, x, y, z, 2]
        if self.prospective is not True:
            print(f'Generating undersampling mask using seed={self.seed}')
            while True:
                try:
                    @timeout(5)  # Timeout set to n seconds
                    def generate_mask():
                        if self.mask == 'poisson2d':
                            return undersample_mask(kspace.shape[1:4], r=self.acc_rate, slope=self.slope,
                                                type='poiss', n_dim=3, calib=self.calib, seed=self.seed), self.calib
                        elif self.mask == 'poisson1d':
                            return undersample_mask(kspace.shape[1:4], r=self.acc_rate, slope=self.slope,
                                                    type='poiss', n_dim=1, calib=self.calib, seed=self.seed), self.calib[0]
                        else:
                            n_x, n_y, n_z = kspace.shape[1:4]
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

            mask = torch.stack([torch.stack([to_tensor(mask)] * kspace.shape[0], dim=0)] * 2, dim=-1)
        else:
            mask = torch.stack([abs(kspace[..., 0]) > 0] * 2, dim=-1)

        # Preprocess on CPUs
        masked_kspaces = []
        for slb in range(kspace.shape[4]):
            slab = kspace[:, :, :, :, slb]
            masked_kspace = slab * mask[..., 0] + 0.

            # Coil compression
            if self.num_compressed_coils is not None:
                compressor = CoilCompressorSVD(out_coils=self.num_compressed_coils, is_3D=False)
                masked_kspace = torch.from_numpy(compressor.compress(masked_kspace.numpy()))

            masked_kspaces.append(masked_kspace)
        masked_kspaces = torch.stack(masked_kspaces, dim=-1)
        mask = mask[:self.num_compressed_coils, ...] if self.num_compressed_coils is not None else mask

        # Estimate maps using ESPIRIRiT
        if self.estimated_maps:
            estimated_maps = torch.zeros_like(masked_kspaces)
            for slb in range(masked_kspaces.shape[4]):
                masked_kspace = masked_kspaces[:, :, :, :, slb]
                masked_kspace = ifftnc(masked_kspace, dim=[1])

                for x in range(masked_kspace.shape[1]):
                    estimated_maps[:, x, :, :, slb] = torch.from_numpy(EspiritCalib(np.asarray(masked_kspace[:, x, ...]),
                                        calib_sz=self.calib, kernel_width=3, thresh=0.02, crop=0.9, show_pbar=False).run())
            estimated_maps = torch.stack([estimated_maps.real, estimated_maps.imag], dim=-1)

        # Save data to h5 file
        os.makedirs(h5_path.parent, exist_ok=True)
        with h5py.File(h5_path, 'w') as hf:
            hf.create_dataset('masked_kspaces', data=masked_kspaces.numpy())
            if self.estimated_maps:
                hf.create_dataset('estimated_maps', data=estimated_maps.numpy())

        return {'masked_kspaces': masked_kspaces,
                'estimated_maps': estimated_maps if self.estimated_maps else 0.,
                'mask': mask,
                'targets': target,
                'data_range': abs(target).max()-abs(target).min(),
                'img_path': str(img_path)
                }