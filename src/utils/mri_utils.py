import itertools
import os
import tempfile
import time

import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import label, generate_binary_structure, binary_fill_holes, binary_dilation, binary_erosion, \
    gaussian_filter
from skimage.filters import threshold_otsu
from skimage.segmentation import find_boundaries
import sigpy.plot as sppl
from scipy.ndimage import convolve1d

from src.data.components.transforms import normalize_01
from src.utils.math import complex_abs_sq
from skimage import exposure, filters, morphology
from fsl.wrappers import flirt, applyxfm
from scipy.linalg import sqrtm, inv
from scipy.ndimage import affine_transform



def apply_affine_to_tensor(tensor: torch.Tensor, affine: np.ndarray, output_shape=None, order=3) -> torch.Tensor:
    """
    Applies a 4x4 affine transformation to a 3D tensor using scipy.ndimage.affine_transform.

    Args:
        tensor (torch.Tensor): Input tensor of shape (1, x, y, z).
        affine (np.ndarray): 4x4 affine matrix.
        output_shape (tuple, optional): Output shape (x, y, z). Defaults to input shape.
        order (int): Interpolation order (0=nearest, 1=linear, 3=cubic).

    Returns:
        torch.Tensor: Transformed tensor of shape (1, x, y, z).
    """
    assert tensor.ndim == 4 and tensor.shape[0] == 1, "Expected shape (1, x, y, z)"
    assert affine.shape == (4, 4), "Affine must be a 4x4 matrix"

    device = tensor.device
    dtype = tensor.dtype

    volume_np = tensor.squeeze(0).cpu().numpy()
    matrix = affine[:3, :3]
    offset = affine[:3, 3]

    if output_shape is None:
        output_shape = volume_np.shape

    transformed = affine_transform(volume_np, matrix=matrix, offset=offset, output_shape=output_shape, order=order)
    return torch.tensor(transformed, dtype=dtype).unsqueeze(0).to(device)

def register_3d_with_flirt_to_midway(output, target):
    """
    Registers both output and target to their midway space using FLIRT.

    Args:
        output (Tensor): shape (1, x, y, z)
        target (Tensor): shape (1, x, y, z)

    Returns:
        output_midway (Tensor): registered to midway
        target_midway (Tensor): registered to midway
        T_output_to_midway (ndarray): 4x4 affine for output → midway
        T_target_to_midway (ndarray): 4x4 affine for target → midway
    """
    device = output.device
    dtype = output.dtype

    output_np = output.squeeze(0).cpu().numpy()
    target_np = target.squeeze(0).cpu().numpy()

    start = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, 'output.nii.gz')
        tgt_path = os.path.join(tmpdir, 'target.nii.gz')
        xfm_path = os.path.join(tmpdir, 'xfm.mat')

        nib.save(nib.Nifti1Image(output_np, affine=np.eye(4)), out_path)
        nib.save(nib.Nifti1Image(target_np, affine=np.eye(4)), tgt_path)

        # Step 1: register output → target (get affine matrix from flirt)
        flirt(src=out_path, ref=tgt_path, omat=xfm_path, dof=6)
        T_ot = np.loadtxt(xfm_path)

        # Step 2: compute midway transform
        T_mid = sqrtm(T_ot)
        T_output_to_midway = inv(T_mid)
        T_target_to_midway = T_mid

        # Step 3: resample using scipy.ndimage.affine_transform
        def apply_affine(vol, T, shape):
            """
            Applies a 4x4 affine transform to a 3D volume using scipy.ndimage.
            """
            matrix = T[:3, :3]
            offset = T[:3, 3]
            return affine_transform(vol, matrix=matrix, offset=offset, output_shape=shape, order=3)

        out_resampled = apply_affine(output_np, T_output_to_midway, output_np.shape)
        tgt_resampled = apply_affine(target_np, T_target_to_midway, target_np.shape)

        output_midway = torch.tensor(out_resampled, dtype=dtype).unsqueeze(0).to(device)
        target_midway = torch.tensor(tgt_resampled, dtype=dtype).unsqueeze(0).to(device)

        print(f"Matrix: {T_mid} | Duration: {time.time() - start:.3f} seconds")

    return output_midway, target_midway, T_output_to_midway, T_target_to_midway


def register_3d_with_flirt(output, target):
    """
    Align a 3D output tensor to the target using FSL FLIRT.

    Args:
        output (Tensor): shape (1, x, y, z)
        target (Tensor): shape (1, x, y, z)

    Returns:
        aligned_output (Tensor): shape (1, x, y, z)
        matrix (np.ndarray): 4x4 FLIRT affine transformation matrix
    """
    device = output.device
    dtype = output.dtype

    output_np = output.squeeze(0).cpu().numpy()  # (x, y, z)
    target_np = target.squeeze(0).cpu().numpy()  # (x, y, z)

    start = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, 'output.nii.gz')
        tgt_path = os.path.join(tmpdir, 'target.nii.gz')
        reg_path = os.path.join(tmpdir, 'aligned.nii.gz')
        mat_path = os.path.join(tmpdir, 'xfm.mat')

        # Save as NIfTI with identity affine
        nib.save(nib.Nifti1Image(output_np, affine=np.eye(4)), out_path)
        nib.save(nib.Nifti1Image(target_np, affine=np.eye(4)), tgt_path)

        # Run FLIRT
        flirt(
            src=out_path,
            ref=tgt_path,
            out=reg_path,
            omat=mat_path,
            dof=6,
        )

        # Load registered image and matrix
        registered = nib.load(reg_path).get_fdata()
        matrix = np.loadtxt(mat_path)

        # Restore to torch tensor with original device and shape (1, x, y, z)
        registered_tensor = torch.tensor(registered, dtype=dtype).unsqueeze(0).to(device)

        print(f"Matrix: {matrix} | Duration: {time.time() - start:.3f} seconds")

    return registered_tensor, matrix


def best_shift_alignment(pred, target, max_shift=2):
    """
    Find the best small shift to align pred to target minimizing MSE.

    Returns:
        best_shift (tuple): (dx, dy)
    Prints:
        Alignment duration in seconds
    """
    assert pred.shape == target.shape and pred.ndim == 4 and pred.shape[0] == 1

    pred = pred[0].max(2)[0]
    target = target[0].max(2)[0]

    best_mse = float('inf')
    best_shift = (0, 0)
    start = time.time()

    for dx, dy in itertools.product(range(-max_shift, max_shift + 1), repeat=2):
        shifted = torch.roll(pred, shifts=(dx, dy), dims=(0, 1))
        mse_val = torch.sum((shifted - target) ** 2 + 1e-8)
        print(f"Shift: {(dx, dy)} | MSE: {mse_val}")
        if mse_val < best_mse:
            best_mse = mse_val
            best_shift = (dx, dy)

    print(f"Best shift: {best_shift} | Alignment duration: {time.time() - start:.3f} seconds")
    return best_shift


def compute_nmse(pred, target):
    """
    Compute NMSE over all elements.

    Args:
        pred (Tensor): predicted values
        target (Tensor): ground truth values

    Returns:
        Tensor: scalar NMSE
    """
    return torch.sum((pred - target) ** 2) / (torch.sum(target ** 2) + torch.finfo(target.dtype).eps)


def create_vessel_mask(im, mask=None, disk_radius=3, thresh=0.5, plot_bool=False):
    """
    Create a vessel mask from an image using morphological and thresholding operations.
    Supports numpy arrays or torch tensors on CPU/GPU.
    Returns output on same device as input if input is torch tensor.

    Parameters:
        im (ndarray or torch.Tensor): Input 2D image
        mask (ndarray or torch.Tensor or None): Binary mask (default: full True mask)
        disk_radius (int): Radius for morphological disk element
        thresh (float): Threshold multiplier
        plot_bool (bool): Whether to display results

    Returns:
        vessel_mask (ndarray or torch.Tensor): Binary mask of vessels, same device/type as input
    """

    input_is_tensor = torch.is_tensor(im)
    device = im.device if input_is_tensor else None

    # Convert tensors to numpy for skimage
    if input_is_tensor:
        im_np = im.detach().cpu().numpy()
    else:
        im_np = im

    if mask is None:
        mask_np = np.ones_like(im_np, dtype=bool)
    else:
        if torch.is_tensor(mask):
            mask_np = mask.detach().cpu().numpy()
        else:
            mask_np = mask

    # Normalize image to [0, 1]
    im_rescaled = exposure.rescale_intensity(im_np, out_range=(0, 1))

    # Morphological background removal
    selem = morphology.disk(disk_radius)
    background = morphology.opening(im_rescaled, selem)
    im2 = exposure.rescale_intensity(im_rescaled - background, out_range=(0, 1))

    # Contrast enhancement
    low, high = np.percentile(im2, [0, 99])
    im3 = exposure.rescale_intensity(im2, in_range=(low, high), out_range=(0, 1))

    # Apply mask
    im4 = im3 * mask_np

    # Global thresholding using Otsu's method (only on masked area)
    level = filters.threshold_otsu(im4[mask_np])
    mask_global = im4 > (level * thresh)

    # Remove small objects (min size 3 pixels)
    vessel_mask_np = morphology.remove_small_objects(mask_global, min_size=3)

    # Apply soft dilation via Gaussian blur and re-threshold
    blurred = gaussian_filter(vessel_mask_np.astype(float), sigma=0.5)
    vessel_mask_np = blurred > 0.15  # Threshold may need tuning

    # Plotting
    if plot_bool:
        # fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        # axes[0].imshow(im_np[:, ::-1], cmap='gray', origin='lower')
        # axes[0].set_title('Input image')
        # axes[0].axis('off')
        #
        # axes[1].imshow(vessel_mask_np[:, ::-1], cmap='gray', origin='lower')
        # axes[1].set_title('Vessel Mask')
        # axes[1].axis('off')
        #
        # axes[2].imshow(im_np[:, ::-1] * vessel_mask_np[:, ::-1], cmap='gray', origin='lower')
        # axes[2].set_title('Masked image')
        # axes[2].axis('off')
        #
        # plt.tight_layout()
        # plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(im_np[:, ::-1], cmap='gray', origin='lower')
        axes[0].set_title('MIP')
        axes[0].axis('off')

        axes[1].imshow(im_np[:, ::-1] * vessel_mask_np[:, ::-1], cmap='gray', origin='lower')
        axes[1].set_title('Vessel-masked')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

    if input_is_tensor:
        # Return tensor on same device & dtype=bool
        return torch.from_numpy(vessel_mask_np).to(device=device, dtype=torch.bool)
    else:
        return vessel_mask_np

def add_gaussian_noise(array, mean_range=(-0.1, 0.1), std_range=(0.01, 0.05), seed=None):
    if seed is not None:
        prev_state = np.random.get_state()
        np.random.seed(seed)

    random_mean = np.random.uniform(*mean_range)
    random_std = np.random.uniform(*std_range)
    # print(random_mean, random_std)
    noise = np.random.normal(random_mean, random_std, array.shape)

    if seed is not None:
        np.random.set_state(prev_state)

    return array + noise

def get_acs(arr, calib):
    size = calib if arr.ndim == 3 else (calib[0], calib[0], calib[1])
    return arr[..., (arr.shape[-3] - size[0]) // 2 : (arr.shape[-3] + size[0]) // 2,
                     (arr.shape[-2] - size[1]) // 2 : (arr.shape[-2] + size[1]) // 2,
                     (arr.shape[-1] - size[2]) // 2 : (arr.shape[-1] + size[2]) // 2] if arr.ndim == 4 else \
           arr[..., (arr.shape[-2] - size[0]) // 2 : (arr.shape[-2] + size[0]) // 2,
                     (arr.shape[-1] - size[1]) // 2 : (arr.shape[-1] + size[1]) // 2]

def apply_gaussian_ratio(outputs, sigma_ratio=1/4, offset=1.0):
    X, Y, Z = outputs.shape[-3:]

    # Define standard deviations for each axis relative to W, H, and D using sigma_ratio
    std_x = X * sigma_ratio  # Standard deviation in x (width)
    std_y = Y * sigma_ratio  # Standard deviation in y (height)
    std_z = X * sigma_ratio  # Standard deviation in z (depth)

    # Create a grid of coordinates for the image
    x, y, z = torch.meshgrid(torch.arange(X, device=outputs.device),
                             torch.arange(Y, device=outputs.device),
                             torch.arange(Z, device=outputs.device),
                             indexing='ij')

    # Calculate the Gaussian distribution with different standard deviations for each axis
    ratio_map = torch.exp(-((x - X // 2) ** 2 / (2 * std_x ** 2) +
                            (y - Y // 2) ** 2 / (2 * std_y ** 2) +
                            (z - Z // 2) ** 2 / (2 * std_z ** 2))).to(outputs.device)

    # Normalize the map and apply the offset
    ratio_map = normalize_01(ratio_map) * (1-offset) + offset

    # Apply the Gaussian filter to the outputs
    return outputs * ratio_map.unsqueeze(0) if outputs.ndim == 4 else outputs * ratio_map


def smooth_concatenate_slabs(image, overlap=0.3):
    """
    Smoothly concatenate slabs of images with a given fraction of overlapping along the z direction.

    Parameters:
    image (np.ndarray or torch.Tensor): Input image with shape [batch, x, y, z, slabs] or [x, y, z, slabs].
    overlap (float): Fraction of the z-direction to overlap between slabs.

    Returns:
    np.ndarray or torch.Tensor: Smoothly concatenated image.
    """
    # Determine whether input is a NumPy array or PyTorch tensor
    is_tensor = torch.is_tensor(image)

    # Check for batch dimension
    is_batched = True
    if len(image.shape) == 4:  # No batch dimension
        image = image.unsqueeze(0) if is_tensor else np.expand_dims(image, axis=0)
        is_batched = False

    batch, x, y, z, slabs = image.shape
    overlap_width = int(z * overlap)

    # Handle zero overlap case
    if overlap_width == 0:
        # Direct concatenation of slabs without overlap
        concatenated_slices = [image[..., :, i] for i in range(slabs)]
        final_image = torch.cat(concatenated_slices, dim=3) if is_tensor else np.concatenate(concatenated_slices,
                                                                                             axis=3)
    else:
        # Prepare list for concatenated slabs with overlap
        concatenated_slices = []

        for i in range(slabs - 1):
            # Current slab and next slab
            current_slab = image[:, :, :, :, i]
            next_slab = image[:, :, :, :, i + 1]

            if i == 0:
                # For the first slab, keep the full non-overlapping part
                non_overlap_current = current_slab[:, :, :, :-overlap_width]
            else:
                # For subsequent slabs, exclude both the initial and final overlaps
                non_overlap_current = current_slab[:, :, :, overlap_width:-overlap_width]

            # Overlapping regions
            overlap_current = current_slab[:, :, :, -overlap_width:]
            overlap_next = next_slab[:, :, :, :overlap_width]

            # Create a smoother blending mask using a cosine function
            blend_mask = (1 - np.cos(np.linspace(0, np.pi, overlap_width))) / 2
            if is_tensor:
                blend_mask = torch.tensor(blend_mask, device=image.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            else:
                blend_mask = blend_mask[np.newaxis, np.newaxis, np.newaxis, :]

            # Blend the overlapping regions symmetrically
            blended_overlap = (overlap_current * (1 - blend_mask) +
                               overlap_next * blend_mask)

            # Append the non-overlapping and blended regions
            concatenated_slices.append(non_overlap_current)
            concatenated_slices.append(blended_overlap)

        # Append the non-overlapping part of the last slab
        concatenated_slices.append(image[:, :, :, :, -1][:, :, :, overlap_width:])

        # Concatenate all slices along the z-axis
        final_image = torch.cat(concatenated_slices, dim=3) if is_tensor else np.concatenate(concatenated_slices,
                                                                                             axis=3)

    # Remove extra batch dimension if the original input had none
    if not is_batched:
        final_image = final_image[0]

    return final_image


def rss(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS).

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    return torch.sqrt((data ** 2).sum(dim))


def rss_complex(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS) for complex inputs.

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    return torch.sqrt(complex_abs_sq(data).sum(dim))


def rss_comb(x, axis=-1):
    if isinstance(x, torch.Tensor):
        return torch.sqrt(torch.sum(torch.abs(x) ** 2, dim=axis))
    else:
        return np.sqrt(np.sum(np.abs(x) ** 2, axis))


def dim_space(data: torch.Tensor):
    """
    Return space dimensions

    Args:
        data: input data

    Returns:
        dimensions
        """
    if data.shape[-1] == 2:
        return [2, 3, 4] if len(data.shape) == 6 else [2, 3]
    else:
        return [2, 3, 4] if len(data.shape) == 5 else [2, 3]


def fftnc(data: torch.Tensor, dim: list = None, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered n-dimensional Fast Fourier Transform.

    Args:
        data: Complex valued input data
        norm: Normalization mode. See ``torch.fft.fft``.

    Returns:
        The FFT of the input.
    """
    if dim == None:
        dim = dim_space(data)

    data = torch.fft.ifftshift(data, dim=dim)
    if data.shape[-1] == 2:
        data = torch.view_as_real(
            torch.fft.fftn(  # type: ignore
                torch.view_as_complex(data), dim=dim, norm=norm
            )
        )
    else:
        data = torch.fft.fftn(data, dim=dim, norm=norm)
    data = torch.fft.fftshift(data, dim=dim)

    return data


def ifftnc(data: torch.Tensor, dim: list = None, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered n-dimensional Inverse Fast Fourier Transform.

    Args:
        data: Complex valued input data
        norm: Normalization mode. See ``torch.fft.ifft``.

    Returns:
        The IFFT of the input.
    """
    if dim == None:
        dim = dim_space(data)

    data = torch.fft.ifftshift(data, dim=dim)
    if data.shape[-1] == 2:
        data = torch.view_as_real(
            torch.fft.ifftn(  # type: ignore
                torch.view_as_complex(data), dim=dim, norm=norm
            )
        )
    else:
        data = torch.fft.ifftn(data, dim=dim, norm=norm)
    data = torch.fft.fftshift(data, dim=dim)

    return data


def simulate_maps(slb_path=None, map_dir=None, slb_data=None, n_coils=32, seed=None):
    if slb_data is None:
        if map_dir == None:
            slb_path = Path(slb_path)
        else:
            slb_path, map_dir = Path(slb_path), Path(map_dir)

            # Create the map directory if it doesn't exist
            os.makedirs(map_dir, exist_ok=True)

            # Load the multi-slab 3D MRA magnitude brain image
            slb_data = np.load(slb_path)

    coil_maps = []
    skull_masks = []
    for slab in range(slb_data.shape[-1]):
        # Estimate the skull mask
        skull_mask = estimate_skull_mask(slb_data[..., slab])
        skull_masks.append(skull_mask)

        # Generate n_coils coil sensitivity maps around the skull edges
        seed_slb = abs(slb_data[..., slab].mean()) if seed == None else (seed + slab) * 1e-9
        coil_maps.append(generate_coil_map(skull_mask, slb_data.shape[:-1], n_coils, seed=seed_slb))

    # Save the coil sensitivity map
    coil_maps = np.stack(coil_maps, axis=-1)

    if not map_dir == None:
        map_filename = f"{slb_path.stem.split('_')[0]}_maps.npy"
        map_path = map_dir / map_filename
        np.save(map_path, coil_maps.astype(np.complex64))
        print(f"Coil maps for {slb_path.name} saved in {map_path}")

    # sppl.ImagePlot(coil_maps[:, :, :, 10, 0], z=0, title='Sensitivity Maps Estimated')
    # sppl.ImagePlot(coil_maps[:, :, :, 10, 4], z=0, title='Sensitivity Maps Estimated')

    return coil_maps, np.stack(skull_masks, axis=-1)


def estimate_skull_masks(slb_data):
    skull_masks = []
    for slab in range(slb_data.shape[-1]):
        # Estimate the skull mask
        skull_mask = estimate_skull_mask(slb_data[..., slab])
        skull_masks.append(skull_mask)
    return np.stack(skull_masks, axis=-1)


def estimate_skull_masks_tensors(slb_data, threshold_scale=0.75, erosion=4, dilation=9):
    skull_masks = []
    device = slb_data.device
    for batch in range(slb_data.shape[0]):
        # Estimate the skull mask
        # skull_mask = estimate_skull_mask(slb_data[batch].cpu().numpy(), threshold_scale=0.75, erosion=3, dilation=7)
        # skull_mask = estimate_skull_mask(slb_data[batch].cpu().numpy(), threshold_scale=0.75, erosion=4, dilation=9) # test
        skull_mask = estimate_skull_mask(slb_data[batch].cpu().numpy(), threshold_scale=threshold_scale, erosion=erosion, dilation=dilation) # prospective
        skull_masks.append(torch.from_numpy(skull_mask))
    return torch.stack(skull_masks, dim=0).to(device)


def estimate_skull_mask(image_data, threshold_scale=1.0, erosion=1, dilation=4):
# def estimate_skull_mask(image_data, threshold_scale=1.0, erosion=5, dilation=15):
    # Perform skull estimation (assuming thresholding or other techniques) of 3D numpy array
    # Use image processing algorithms to estimate the skull mask
    # Here, we use Otsu's thresholding
    threshold_value = threshold_otsu(image_data)
    skull_mask = image_data > threshold_value * threshold_scale

    # plt.imshow(image_data[:, :, 10], cmap='gray', origin='lower')
    # plt.show()
    # plt.imshow(image_data[128, :, :], cmap='gray', origin='lower')
    # plt.show()
    # plt.imshow(skull_mask[:, :, 10], cmap='gray', origin='lower')
    # plt.show()
    # plt.imshow(skull_mask[128, :, :], cmap='gray', origin='lower')
    # plt.show()

    # # Clean the mask by keeping only the largest connected component (the skull)
    # labeled, num_features = label(skull_mask, structure=generate_binary_structure(3, 1))
    # sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
    # max_label = sizes.index(max(sizes)) + 1
    # skull_mask = labeled == max_label

    # set top and bottom layers to be True to enclose the head region
    top, btm = skull_mask[:, :, -1], skull_mask[:, :, 0]
    skull_mask[:, :, -1] = np.ones_like(skull_mask[:, :, -1], dtype=bool)
    skull_mask[:, :, 0] = np.ones_like(skull_mask[:, :, 0], dtype=bool)

    # Fill the spaces within the outer boundaries of skull_mask
    for z in range(skull_mask.shape[-1]):
        skull_mask[:, :, z] = binary_fill_holes(skull_mask[:, :, z])
    for x in range(skull_mask.shape[0]):
        skull_mask[x, :, :] = binary_fill_holes(skull_mask[x, :, :])
    for y in range(skull_mask.shape[1]):
        skull_mask[:, y, :] = binary_fill_holes(skull_mask[:, y, :])
    for z in range(skull_mask.shape[-1]):
        skull_mask[:, :, z] = binary_fill_holes(skull_mask[:, :, z])

    # set top and bottom layers to original values
    skull_mask[:, :, -1], skull_mask[:, :, 0] = top, btm

    # Perform dilation and erosion for refining the filling
    skull_mask = binary_erosion(skull_mask, iterations=erosion)
    skull_mask = binary_dilation(skull_mask, iterations=dilation)

    # plt.imshow(skull_mask[:, :, 10], cmap='gray', origin='lower')
    # plt.show()
    # plt.imshow(skull_mask[128, :, :], cmap='gray', origin='lower')
    # plt.show()

    return skull_mask


def generate_coil_map(skull_mask, image_shape, n_coils, seed=None):
    # Generate n coil sensitivity maps around the skull edges
    # Use Gaussian distributions centered around the skull edges

    # Find edges of the skull mask
    skull_edges = find_boundaries(skull_mask, mode='outer')

    # plt.imshow(skull_edges[:, :, 10], cmap='gray', origin='lower')
    # plt.show()
    # plt.imshow(skull_edges[:, :, 15], cmap='gray', origin='lower')
    # plt.show()
    # plt.imshow(skull_edges[:, :, 5], cmap='gray', origin='lower')
    # plt.show()
    # plt.imshow(skull_edges[64, :, :], cmap='gray', origin='lower')
    # plt.show()
    # plt.imshow(skull_edges[:, 250, :], cmap='gray', origin='lower')
    # plt.show()

    # Find coordinates of skull edges
    edge_coords = np.array(np.where(skull_edges)).T

    # Define variance for the Gaussian distribution
    std = image_shape[0] / 4  # You may adjust this value for the width of the Gaussian

    x, y, z = np.meshgrid(np.arange(image_shape[0]), np.arange(image_shape[1]), np.arange(image_shape[2]))

    prev_state = np.random.get_state()

    maps = []
    maps_phs = []
    for i in range(n_coils):
        # Generate a single coil sensitivity map around the skull edges
        np.random.seed(i + int(seed * 1e+10))

        # Randomly select a coordinate from skull edges
        center_idx = np.random.randint(len(edge_coords))
        center_x, center_y, center_z = edge_coords[center_idx]

        # Define different standard deviations for each axis
        std_rand_x = np.random.normal(std, std / 5.1)
        std_rand_y = np.random.normal(std, std / 5.2)
        std_rand_z = np.random.normal(std, std / 5.3)

        # Define the angle of rotation in radians
        angle = np.random.rand() * 2 * np.pi  # Random angle between 0 and 2*pi
        # angle = 0 # CMR

        # # for comparison
        # std_rand_x = std
        # std_rand_y = std
        # std_rand_z = std
        # angle = 0.

        # Calculate the rotated coordinates in the x-y plane
        rotated_x = (x - center_x) * np.cos(angle) - (y - center_y) * np.sin(angle) + center_x
        rotated_y = (x - center_x) * np.sin(angle) + (y - center_y) * np.cos(angle) + center_y
        rotated_z = z  # No rotation in the z-axis
        rotated_x, rotated_y, rotated_z = rotated_x.astype(np.float32), rotated_y.astype(np.float32), rotated_z.astype(
            np.float32)

        # Calculate the Gaussian distribution with different standard deviations for each axis
        map = np.exp(-((rotated_x - center_x) ** 2 / (2 * std_rand_x ** 2) +
                       (rotated_y - center_y) ** 2 / (2 * std_rand_y ** 2) +
                       (rotated_z - center_z) ** 2 / (2 * std_rand_z ** 2)))

        # # Add sinusoidal patterns along each axis
        # freq_x = np.random.uniform(0.501 / image_shape[0], 1.501 / image_shape[0])  # Frequency along x-axis
        # freq_y = np.random.uniform(0.502 / image_shape[1], 1.502 / image_shape[1])  # Frequency along y-axis
        # freq_z = np.random.uniform(0.0103 / image_shape[2], 0.0503 / image_shape[2])  # Frequency along z-axis
        #
        # sin_x = np.sin(2 * np.pi * freq_x * rotated_x)
        # sin_y = np.sin(2 * np.pi * freq_y * rotated_y)
        # sin_z = np.sin(2 * np.pi * freq_z * rotated_z)
        #
        # # Combine sinusoidal patterns with Gaussian distribution
        # sinusoids = sin_x + sin_y + sin_z
        # sinusoids = normalize_01(sinusoids)
        # map *= (1.0 + 1.0 * (sinusoids - 0.5))

        # Add Fourier
        # map = add_random_noise_fourier(map, std_ratio=0.1, std_scale=0.0075)
        # map = add_random_noise_fourier(map, std_ratio=0.001, std_scale=1.0)
        # map = add_random_noise_fourier(map, std_ratio=np.random.uniform(0, 0.25), std_scale=0.0075)
        # map = add_random_noise_fourier(map, std_ratio=np.random.uniform(0, 0.0001), std_scale=1.0)
        # map = add_random_noise_fourier(map, # CMR
        #                                std_ratio=[np.random.uniform(0, 0.25),
        #                                           np.random.uniform(0, 0.0001)],
        #                                std_scale=[0.0075, 1.0])
        map = add_random_noise_fourier(
            map * np.sqrt(2) * np.exp(1j * np.ones_like(map) * np.random.uniform(-np.pi, np.pi)),
            std_ratio=[np.random.uniform(0, 0.25),
                       np.random.uniform(0, 0.0001)],
            std_scale=[0.0075, 1.0])
        map_phs = np.angle(map)
        maps_phs.append(map_phs.astype(np.float32))

        map = np.abs(map)
        maps.append(map.astype(np.float32) * np.random.uniform(0.1, 1.0))

    maps = np.stack(maps, axis=0)

    # std = image_shape[0] / 3
    # for i in range(n_coils):
    #     # Generate a single coil sensitivity map around the skull edges
    #     np.random.seed(-i + int(seed * 1e+10) - 1)
    #
    #     # # Randomly select a coordinate from skull edges
    #     # center_idx = np.random.randint(len(edge_coords))
    #     # center_x, center_y, center_z = edge_coords[center_idx]
    #     #
    #     # # Define different standard deviations for each axis
    #     # std_rand_x = np.random.normal(std, std / 5.1)
    #     # std_rand_y = np.random.normal(std, std / 5.2)
    #     # std_rand_z = np.random.normal(std, std / 5.3)
    #     #
    #     # # Define the angle of rotation in radians
    #     # angle = np.random.rand() * 2 * np.pi  # Random angle between 0 and 2*pi
    #     #
    #     # # # for comparison
    #     # # std_rand_x = std
    #     # # std_rand_y = std
    #     # # std_rand_z = std
    #     # # angle = 0.
    #     #
    #     # # Calculate the rotated coordinates in the x-y plane
    #     # rotated_x = (x - center_x) * np.cos(angle) - (y - center_y) * np.sin(angle) + center_x
    #     # rotated_y = (x - center_x) * np.sin(angle) + (y - center_y) * np.cos(angle) + center_y
    #     # rotated_z = z  # No rotation in the z-axis
    #     # rotated_x, rotated_y, rotated_z = rotated_x.astype(np.float32), rotated_y.astype(np.float32), rotated_z.astype(np.float32)
    #     #
    #     # # Calculate the Gaussian distribution with different standard deviations for each axis
    #     # map_phs = np.exp(-((rotated_x - center_x) ** 2 / (2 * std_rand_x ** 2) +
    #     #                (rotated_y - center_y) ** 2 / (2 * std_rand_y ** 2) +
    #     #                (rotated_z - center_z) ** 2 / (2 * std_rand_z ** 2)))
    #     #
    #     # # Add sinusoidal patterns along each axis
    #     # freq_x = np.random.uniform(0.101 / image_shape[0], 0.2501 / image_shape[0])  # Frequency along x-axis
    #     # freq_y = np.random.uniform(0.102 / image_shape[1], 0.2502 / image_shape[1])  # Frequency along y-axis
    #     # freq_z = np.random.uniform(0.001 / image_shape[2], 0.002 / image_shape[2])  # Frequency along z-axis
    #     #
    #     # sin_x = np.sin(2 * np.pi * freq_x * rotated_x)
    #     # sin_y = np.sin(2 * np.pi * freq_y * rotated_y)
    #     # sin_z = np.sin(2 * np.pi * freq_z * rotated_z)
    #     #
    #     # # Combine sinusoidal patterns with Gaussian distribution
    #     # sinusoids = sin_x + sin_y + sin_z
    #     # # sinusoids = (sinusoids - sinusoids.min()) / (sinusoids.max() - sinusoids.min()) - 0.5
    #     # map_phs = normalize_01(map_phs) - 0.5
    #     # # map_phs = 0.25 * map_phs + sinusoids
    #     # map_phs = sinusoids * (1.0 + 0.25 * map_phs)
    #
    #     # Add Fourier
    #     # map_phs = generate_Fourier_truncation(image_shape, truncation=2)
    #     # map_phs = add_random_noise_fourier(map_phs, std_ratio=0.005, std_scale=0.1)
    #     # map_phs = add_random_noise_fourier(map_phs, std_ratio=0.001, std_scale=1.0)
    #     # map_phs = add_random_noise_fourier(map_phs, std_ratio=np.random.uniform(0, 0.005), std_scale=0.1)
    #     # map_phs = add_random_noise_fourier(map_phs, std_ratio=np.random.uniform(0, 0.001), std_scale=1.0)
    #     map_phs = np.ones(image_shape[0:3], dtype=np.float32) + 1j * np.ones(image_shape[0:3], dtype=np.float32)
    #     # map_phs = np.ones_like(map_phs) * np.exp(1j * map_phs)
    #     map_phs = add_random_noise_fourier(map_phs,
    #                                        std_ratio=[np.random.uniform(0, 3.0),
    #                                                   np.random.uniform(0, 0.001),
    #                                                   np.random.uniform(0, 0.00025)],
    #                                        std_scale=[0.0075, 0.1, 1.0])
    #     map_phs = np.angle(map_phs)
    #
    #     # Flip values outside the range [-1, 1]
    #     # map_phs = np.where(np.abs(map_phs) > 1, -np.sign(map_phs) * (2 - np.abs(map_phs)), map_phs)
    #
    #     # maps_phs.append(map_phs.astype(np.float32) * np.pi)
    #     maps_phs.append(map_phs.astype(np.float32))

    maps_phs = np.stack(maps_phs, axis=0)

    np.random.set_state(prev_state)

    # Normalise
    par = 10
    # sppl.ImagePlot(maps[:, :, :, par], z=0, title='Sensitivity Maps - Magnitude')
    # sppl.ImagePlot(maps_phs[:, :, :, par], z=0, title='Sensitivity Maps Estimated - Phase')
    # print(f'Mag: [{np.min(maps[:, :, :, par])}, {np.max(maps[:, :, :, par])}]')
    # print(f'Phs: [{np.min(maps_phs[:, :, :, par])}, {np.max(maps_phs[:, :, :, par])}]')
    eps = np.finfo(np.float32).eps
    rss = rss_comb(maps, axis=0) + eps
    maps /= np.stack([rss] * maps.shape[0], axis=0)
    # # for comparison
    # rss_phs = rss_comb(maps_phs, axis=0) + eps
    # maps_phs /= np.stack([rss_phs] * maps_phs.shape[0], axis=0)
    # maps_phs = (normalize_01(maps_phs) * 2. - 1.) * np.pi

    # sppl.ImagePlot(maps[:, :, :, par], z=0, title='Sensitivity Maps - Magnitude')
    # sppl.ImagePlot(maps_phs[:, :, :, par], z=0, title='Sensitivity Maps Estimated - Phase')
    # print(f'Mag: [{np.min(maps[:, :, :, par])}, {np.max(maps[:, :, :, par])}]')
    # print(f'Phs: [{np.min(maps_phs[:, :, :, par])}, {np.max(maps_phs[:, :, :, par])}]')
    # sppl.ImagePlot(rss_phs[:, :, par], title='Sensitivity Maps - Phase RSS')
    # print(f'Phs RSS: [{np.min(rss_phs[:, :, par])}, {np.max(rss_phs[:, :, par])}]')

    # create complex maps
    maps = (maps + eps) * np.exp(1j * maps_phs) + 0.
    # masked
    # maps = maps * skull_mask * np.exp(1j * maps_phs * skull_mask) + 0.
    # sppl.ImagePlot(abs(maps[:, :, :, par]), z=0, title='Sensitivity Maps - Magnitude')
    # sppl.ImagePlot(np.angle(maps[:, :, :, par]), z=0, title='Sensitivity Maps Estimated - Phase')
    # sppl.ImagePlot(maps[:, :, :, par].real, z=0, title='Sensitivity Maps - Real')
    # sppl.ImagePlot(maps[:, :, :, par].imag, z=0, title='Sensitivity Maps Estimated - Imag')
    # print(abs(maps[:, :, :, par]).max(), abs(maps[:, :, :, par]).min())
    # print(np.angle(maps[:, :, :, par]).max(), np.angle(maps[:, :, :, par]).min())
    # print(maps[:, :, :, par].real.max(), maps[:, :, :, par].real.min())
    # print(maps[:, :, :, par].imag.max(), maps[:, :, :, par].imag.min())
    # rss = rss_comb(maps, axis=0) + eps
    # sppl.ImagePlot(rss[:, :, par], title='Sensitivity Maps - RSS')
    # print(f'RSS: [{np.min(rss[:, :, par])}, {np.max(rss[:, :, par])}]')

    return maps.astype(np.complex64)


def generate_Fourier_truncation(shape, truncation=3):
    """
    Generate randomly generated Fourier truncation of complex white Gaussian noise.

    Parameters:
        shape (tuple): Shape of the output phase profile.
        truncation (int): Truncation radius for Fourier truncation.

    Returns:
        np.ndarray: Phase profile with the specified shape.
    """
    # Generate complex white Gaussian noise
    noise_complex = np.random.normal(0, 1, size=shape) + 1j * np.random.normal(0, 1, size=shape)

    # Compute Fourier transform of the noise and shift to center low frequencies
    noise_fft_shifted = np.fft.fftshift(np.fft.fftn(noise_complex))

    # Define the bounds for selecting the central cube
    start_idx = [(freq_dim - truncation) // 2 for freq_dim in noise_fft_shifted.shape]
    end_idx = [start + truncation for start in start_idx]

    # Perform Fourier truncation by selecting the centre
    noise_fft_shifted_truncated = np.zeros_like(noise_fft_shifted)
    central_cube = tuple(slice(start, end) for start, end in zip(start_idx, end_idx))
    noise_fft_shifted_truncated[central_cube] = noise_fft_shifted[central_cube]

    # Compute inverse Fourier transform to get the phase profile and unshift
    truncation = np.fft.ifftn(np.fft.ifftshift(noise_fft_shifted_truncated))

    return truncation


def as_list(input):
    return input if isinstance(input, list) else [input]

def add_random_noise_fourier(array_3d, std_ratio, std_scale):
    """
    Add random points with random values in the Fourier domain of a 3D numpy array.

    Parameters:
        array_3d (np.ndarray): The input 3D numpy array.
        num_points (int): The number of random points to add in the Fourier domain.
        max_ratio (float): The maximum ratio of value for the random values to be added.

    Returns:
        np.ndarray: The modified 3D numpy array with random points added in the Fourier domain.
    """
    # Compute the Fourier transform of the input array
    fft_array = np.fft.fftshift(np.fft.fftn(array_3d))
    # plt.imshow(np.abs(array_3d[:, :, 12]), cmap='gray', origin='lower')
    # plt.show()
    # plt.imshow(np.angle(array_3d[:, :, 12]), cmap='gray', origin='lower')
    # plt.show()
    # plt.imshow(abs(fft_array[:, :, 12]) ** 0.2, cmap='gray', origin='lower')
    # plt.show()
    # plt.imshow(abs(fft_array[128, :, :]) ** 0.2, cmap='gray', origin='lower')
    # plt.show()

    std_ratios = as_list(std_ratio)
    std_scales = as_list(std_scale)
    for std_ratio, std_scale in zip(std_ratios, std_scales):
        # Generate random noise with the same shape as the Fourier domain
        noise = (np.random.normal(scale=np.max(fft_array.real) * std_ratio, size=fft_array.shape) +
                 1j * np.random.normal(scale=np.max(fft_array.imag) * std_ratio, size=fft_array.shape))

        # Create a meshgrid of coordinates centered at the origin
        shape = fft_array.shape
        x, y, z = np.meshgrid(
            np.arange(-shape[0] // 2, shape[0] // 2),
            np.arange(-shape[1] // 2, shape[1] // 2),
            np.arange(-shape[2] // 2, shape[2] // 2),
            indexing='ij'
        )

        # Create a Gaussian kernel
        # scale_map = np.exp(-(x ** 2 + y ** 2 + z ** 2) / (2 * (shape[0]/2 * std_scale) ** 2))
        scale_map = np.exp(-(x ** 2 / (2 * (shape[0] / 2 * std_scale) ** 2) +
                             y ** 2 / (2 * (shape[1] / 2 * std_scale) ** 2) +
                             z ** 2 / (2 * (shape[2] / 2 * std_scale) ** 2)))

        # Normalize the scale map
        scale_map /= np.max(scale_map)

        # Add the noise to the Fourier domain
        fft_array += noise * scale_map

    # Compute the inverse Fourier transform to obtain the modified array
    modified_array = np.fft.ifftn(np.fft.ifftshift(fft_array))
    # plt.imshow(np.abs(modified_array[:, :, 12]), cmap='gray', origin='lower')
    # plt.show()
    # plt.imshow(np.angle(modified_array[:, :, 12]), cmap='gray', origin='lower')
    # plt.show()
    # plt.imshow(abs(fft_array[:, :, 12]) ** 0.2, cmap='gray', origin='lower')
    # plt.show()
    # plt.imshow(abs(fft_array[128, :, :]) ** 0.2, cmap='gray', origin='lower')
    # plt.show()

    return modified_array

def add_random_noise_fourier_dc(array_3d, std_ratio, std_scale):
    """
    Add random points with random values in the Fourier domain of a 3D numpy array (DC removed).

    Parameters:
        array_3d (np.ndarray): The input 3D numpy array.
        num_points (int): The number of random points to add in the Fourier domain.
        max_ratio (float): The maximum ratio of value for the random values to be added.

    Returns:
        np.ndarray: The modified 3D numpy array with random points added in the Fourier domain.
    """
    # Compute the Fourier transform of the input array
    fft_array = np.fft.fftshift(np.fft.fftn(array_3d))
    fft_array_dc = fft_array.copy()
    fft_array_dc[tuple(s // 2 for s in fft_array.shape[:3])] = 0
    # plt.imshow(np.abs(array_3d[:, :, 12]), cmap='gray', origin='lower')
    # plt.show()
    # plt.imshow(np.angle(array_3d[:, :, 12]), cmap='gray', origin='lower')
    # plt.show()
    # plt.imshow(abs(fft_array[:, :, 12]) ** 0.2, cmap='gray', origin='lower')
    # plt.show()
    # plt.imshow(abs(fft_array[128, :, :]) ** 0.2, cmap='gray', origin='lower')
    # plt.show()

    std_ratios = as_list(std_ratio)
    std_scales = as_list(std_scale)
    for std_ratio, std_scale in zip(std_ratios, std_scales):
        # Generate random noise with the same shape as the Fourier domain
        noise = (np.random.normal(scale=np.max(fft_array_dc.real) * std_ratio, size=fft_array.shape) +
                 1j * np.random.normal(scale=np.max(fft_array_dc.imag) * std_ratio, size=fft_array.shape))

        # Create a meshgrid of coordinates centered at the origin
        shape = fft_array.shape
        x, y, z = np.meshgrid(
            np.arange(-shape[0] // 2, shape[0] // 2),
            np.arange(-shape[1] // 2, shape[1] // 2),
            np.arange(-shape[2] // 2, shape[2] // 2),
            indexing='ij'
        )

        # Create a Gaussian kernel
        # scale_map = np.exp(-(x ** 2 + y ** 2 + z ** 2) / (2 * (shape[0]/2 * std_scale) ** 2))
        scale_map = np.exp(-(x ** 2 / (2 * (shape[0] / 2 * std_scale) ** 2) +
                             y ** 2 / (2 * (shape[1] / 2 * std_scale) ** 2) +
                             z ** 2 / (2 * (shape[2] / 2 * std_scale) ** 2)))

        # Normalize the scale map
        scale_map /= np.max(scale_map)

        # Add the noise to the Fourier domain
        fft_array += noise * scale_map

    # Compute the inverse Fourier transform to obtain the modified array
    modified_array = np.fft.ifftn(np.fft.ifftshift(fft_array))
    # plt.imshow(np.abs(modified_array[:, :, 12]), cmap='gray', origin='lower')
    # plt.show()
    # plt.imshow(np.angle(modified_array[:, :, 12]), cmap='gray', origin='lower')
    # plt.show()
    # plt.imshow(abs(fft_array[:, :, 12]) ** 0.2, cmap='gray', origin='lower')
    # plt.show()
    # plt.imshow(abs(fft_array[128, :, :]) ** 0.2, cmap='gray', origin='lower')
    # plt.show()

    return modified_array

def simulate_phase(slb_path=None, phs_dir=None, slb_data=None, seed=None, skull_masks=None):
    if slb_data is None:
        if phs_dir == None:
            slb_path = Path(slb_path)
        else:
            slb_path, phs_dir = Path(slb_path), Path(phs_dir)

            # Create the phase directory if it doesn't exist
            os.makedirs(phs_dir, exist_ok=True)

            # Load the multi-slab MRA magnitude NPY brain image
            slb_data = np.load(slb_path)

    eps = np.finfo(np.float32).eps

    prev_state = np.random.get_state()
    phase_maps = []
    for slb in range(slb_data.shape[-1]):
        np.random.seed(seed * (slb + 1)) if seed is not None else None

        # # Get the image shape and determine the centres for the positive and negative Gaussian
        # x_center_frontal_lobe = slb_data.shape[0] * 3 // 4
        #
        # # Generate random centers & std using Gaussian probability distribution
        # center_frontal = (np.random.normal(x_center_frontal_lobe, slb_data.shape[0] // 30),
        #                    np.random.normal(slb_data.shape[1] // 2, slb_data.shape[1] // 50),
        #                    np.random.normal(slb_data.shape[2] // 2, slb_data.shape[2] // 50))
        #
        # # center_positive = (np.random.normal(slb_data.shape[0] // 2, slb_data.shape[0] // 30),
        # #                    np.random.normal(slb_data.shape[1] // 2, slb_data.shape[1] // 50),
        # #                    np.random.normal(slb_data.shape[2] // 2, slb_data.shape[2] // 50))
        #
        # center_negative = (np.random.normal(slb_data.shape[0] // 2, slb_data.shape[0] // 40),
        #                    np.random.normal(slb_data.shape[1] // 2, slb_data.shape[1] // 60),
        #                    np.random.normal(slb_data.shape[2] // 2, slb_data.shape[2] // 60))
        #
        #
        # std_frontal = np.random.normal(slb_data.shape[0] // 15, slb_data.shape[0] // 100)
        # # std_positive = np.random.normal(slb_data.shape[0] // 4.5, 15)
        # std_negative = np.random.normal(slb_data.shape[0] // 5, slb_data.shape[0] // 100)
        #
        # # Define different standard deviations for each axis
        # std_frontal_x = np.random.normal(std_frontal, std_frontal / 5.1)
        # std_frontal_y = np.random.normal(std_frontal, std_frontal / 5.2)
        # std_frontal_z = np.random.normal(std_frontal, std_frontal / 5.3)
        # std_negative_x = np.random.normal(std_negative * 1.2, std_negative / 20.1)
        # std_negative_y = np.random.normal(std_negative * 1.1, std_negative / 20.2)
        # std_negative_z = np.random.normal(std_negative, std_negative / 20.3)
        #
        # # Define the angle of rotation in radians
        # angle_frontal = np.random.rand(2)[0] * 2 * np.pi  # Random angle between 0 and 2*pi
        # angle_negative = np.random.rand(2)[1] * 0.1 * np.pi
        #
        # # Create a grid of coordinates for the image
        # x, y, z = np.meshgrid(np.arange(slb_data.shape[0]), np.arange(slb_data.shape[1]), np.arange(slb_data.shape[2]),
        #                       indexing='ij')
        #
        # # Calculate the rotated coordinates in the x-y plane
        # rotated_frontal_x = (x - center_frontal[0]) * np.cos(angle_frontal) - (y - center_frontal[1]) * np.sin(angle_frontal) + center_frontal[0]
        # rotated_frontal_y = (x - center_frontal[0]) * np.sin(angle_frontal) + (y - center_frontal[1]) * np.cos(angle_frontal) + center_frontal[1]
        # rotated_frontal_z = z  # No rotation in the z-axis
        # rotated_negative_x = (x - center_negative[0]) * np.cos(angle_negative) - (y - center_negative[1]) * np.sin(angle_negative) + center_negative[0]
        # rotated_negative_y = (x - center_negative[0]) * np.sin(angle_negative) + (y - center_negative[1]) * np.cos(angle_negative) + center_negative[1]
        # rotated_negative_z = z  # No rotation in the z-axis
        # rotated_frontal_x, rotated_frontal_y, rotated_frontal_z = rotated_frontal_x.astype(np.float32), rotated_frontal_y.astype(np.float32), rotated_frontal_z.astype(np.float32)
        # rotated_negative_x, rotated_negative_y, rotated_negative_z = rotated_negative_x.astype(np.float32), rotated_negative_y.astype(np.float32), rotated_negative_z.astype(np.float32)
        #
        # # Calculate the Gaussian distribution with different standard deviations for each axis
        # phase_frontal = np.exp(-((rotated_frontal_x - center_frontal[0]) ** 2 / (2 * std_frontal_x ** 2) +
        #                          (rotated_frontal_y - center_frontal[1]) ** 2 / (2 * std_frontal_y ** 2) +
        #                          (rotated_frontal_z - center_frontal[2]) ** 2 / (2 * std_frontal_z ** 2)))
        # phase_negative = -np.exp(-((rotated_negative_x - center_negative[0]) ** 2 / (2 * std_negative_x ** 2) +
        #                          (rotated_negative_y - center_negative[1]) ** 2 / (2 * std_negative_y ** 2) +
        #                          (rotated_negative_z - center_negative[2]) ** 2 / (2 * std_negative_z ** 2)))
        #
        # # Combine the positive and negative Gaussians to create the final phase map
        # phase_map = phase_frontal + phase_negative
        # # max_val = phase_map.max()
        # # phase_map += (1.25 - max_val) * phase_frontal

        # # Add sinusoidal patterns along each axis
        # freq_x = np.random.uniform(0.501 / slb_data.shape[0], 2.501 / slb_data.shape[0])  # Frequency along x-axis
        # freq_y = np.random.uniform(0.502 / slb_data.shape[1], 2.502 / slb_data.shape[1])  # Frequency along y-axis
        # freq_z = np.random.uniform(0.103 / slb_data.shape[2], 1.003 / slb_data.shape[2])  # Frequency along z-axis
        #
        # sin_x = np.sin(2 * np.pi * freq_x * rotated_frontal_x)
        # sin_y = np.sin(2 * np.pi * freq_y * rotated_frontal_y)
        # sin_z = np.sin(2 * np.pi * freq_z * rotated_frontal_z)
        #
        # # Combine sinusoidal patterns with Gaussian distribution
        # sinusoids = sin_x + sin_y + sin_z
        # sinusoids = (sinusoids - sinusoids.min()) / (sinusoids.max() - sinusoids.min()) - 0.5
        # phase_map = 0.75 * phase_map + np.random.uniform(0.5, 1.0) * sinusoids
        #
        # # Add noise
        # max_val = phase_map.max()
        # noise = np.random.normal(0, max_val/50, phase_map.shape).astype(np.float32)
        # phase_map = phase_map + noise
        # # noise_bg = np.random.normal(0, max_val/2, phase_map.shape).astype(np.float32)
        # # noise_bg[abs(phase_negative) > max_val/5] = 0
        # # phase_map = phase_map + noise + noise_bg
        #
        # # Add high-frequency
        # phase_map = np.ones_like(phase_map) * np.exp(1j * phase_map)
        # phase_map = add_random_noise_fourier(phase_map, std_ratio=0.0005, std_scale=0.1)
        # phase_map = add_random_noise_fourier(phase_map, std_ratio=0.0001, std_scale=1.0)
        # phase_map = np.angle(phase_map)
        #
        # if skull_masks is not None:
        #     # skull_mask = np.stack([binary_erosion(skull_masks[..., z, slb], iterations=3) for z in range(skull_masks.shape[2])], axis=2)
        #     # phase_map *= skull_mask
        #     phase_map *= skull_masks[..., slb]
        #     # noise_bg = np.random.normal(0, max_val/2, phase_map.shape).astype(np.float32)
        #     noise_bg = np.random.uniform(phase_map.min(), max_val, phase_map.shape).astype(np.float32)
        #     noise_bg[abs(phase_map) > 0] = 0
        #     phase_map += noise_bg
        #
        # # Flip values outside the range [-1, 1]
        # # phase_map = np.where(np.abs(phase_map) > 1, -np.sign(phase_map) * (2 - np.abs(phase_map)), phase_map)
        #
        # # Normalize the phase map
        # # phase_map = (phase_map - phase_map.min()) / (phase_map.max() - phase_map.min())
        #
        # # #  for comparison
        # # phase_map = normalize_01(sinusoids) * 2 - 1
        #
        # # Scale the phase values to span from -π to π
        # # phase_map = phase_map * 2 * np.pi - np.pi
        # phase_map = phase_map * np.pi
        # # print(phase_map.max(), phase_map.min())

        # # Add Fourier
        # phase_map = generate_Fourier_truncation(slb_data.shape[0:3], truncation=np.random.randint(2,5)) # LORAKS
        # # phase_map = add_random_noise_fourier(phase_map, std_ratio=0.005, std_scale=0.1)
        # # phase_map = add_random_noise_fourier(phase_map, std_ratio=0.001, std_scale=1.0)
        # phase_map = np.ones_like(phase_map) * np.exp(1j * phase_map * np.random.uniform(1.0, np.pi))
        # # phase_map = (slb_data[...,slb] + eps) * np.exp(1j * phase_map * np.random.uniform(1.0, np.pi))
        # # phase_map = add_random_noise_fourier(phase_map, std_ratio=np.random.uniform(0, 0.5), std_scale=0.01)
        # # phase_map = add_random_noise_fourier(phase_map, std_ratio=np.random.uniform(0, 0.005), std_scale=0.1)
        # # phase_map = add_random_noise_fourier(phase_map, std_ratio=np.random.uniform(0, 0.001), std_scale=1.0)
        # # phase_map = add_random_noise_fourier(phase_map,
        # #                                      std_ratio=[np.random.uniform(0, 0.5),
        # #                                                 np.random.uniform(0, 0.005),
        # #                                                 np.random.uniform(0, 0.001)],
        # #                                      std_scale=[0.01, 0.1, 1.0])
        # phase_map = add_random_noise_fourier(phase_map,
        #                                      std_ratio=[np.random.uniform(0, 0.5),
        #                                                 np.random.uniform(0, 0.001),
        #                                                 np.random.uniform(0, 0.0001)],
        #                                      std_scale=[0.01, 0.1, 1.0])
        # phase_map = np.angle(phase_map)
        #
        # phase_maps.append(phase_map.astype(np.float32))

        # phase_map = np.ones(slb_data.shape[:3], dtype=np.float32)
        # phase_map = add_random_noise_fourier(phase_map,
        #                                      std_ratio=[1.0],
        #                                      std_scale=[1/3])
        phase_map = (slb_data[..., slb] + eps)
        # # For TOF
        # phase_map = add_random_noise_fourier(phase_map,
        #                                      std_ratio=[np.random.uniform(1.0, 3.0),
        #                                                 np.random.uniform(0, 0.01),
        #                                                 np.random.uniform(0, 0.0001)],
        #                                      std_scale=[np.random.uniform(0.005, 0.025),
        #                                                 np.random.uniform(0.025, 0.25),
        #                                                 1.0])
        # For ASL
        phase_map = add_random_noise_fourier_dc(phase_map,
                                                std_ratio=[np.random.uniform(0.5, 0.75)],
                                                std_scale=[1.0])
        phase_map = np.angle(phase_map)

        phase_maps.append(phase_map.astype(np.float32))

    # # Same phase maps for all slabs
    # phase_maps = np.stack([phase_map] * slb_data.shape[-1], axis=-1)

    phase_maps = np.stack(phase_maps, axis=-1)

    np.random.set_state(prev_state)

    # Save the simulated phase map
    if not phs_dir == None:
        phs_filename = f"{slb_path.stem.split('_')[0]}_phase.npy"
        phs_path = phs_dir / phs_filename
        np.save(phs_path, phase_maps.astype(np.float32))
        print(f"Phase map for {slb_path.name} saved in {phs_path}")

    # plt.imshow(phase_maps[:, :, 12, 1], cmap='gray', origin='lower')
    # plt.show()

    return phase_maps


def unnormalise_target(data=None, seed=None):
    prev_state = np.random.get_state()
    targets = []
    for slb in range(data.shape[-1]):
        np.random.seed(seed * (slb + 1)) if seed is not None else None

        center = (np.random.normal(data.shape[0] // 2, data.shape[0] // 50),
                  np.random.normal(data.shape[1] // 2, data.shape[1] // 50),
                  np.random.normal(data.shape[2] // 2, data.shape[2] // 50))

        std = np.random.normal(data.shape[0] // 4.5, data.shape[0] // 20)

        # Define different standard deviations for each axis
        std_x = np.random.normal(std, std / 15.1)
        std_y = np.random.normal(std, std / 15.2)
        std_z = np.random.normal(std, std / 15.3)

        # Define the angle of rotation in radians
        angle = np.random.rand(2)[1] * 2 * np.pi

        # Create a grid of coordinates for the image
        x, y, z = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]), np.arange(data.shape[2]),
                              indexing='ij')

        # Calculate the rotated coordinates in the x-y plane
        rotated_x = (x - center[0]) * np.cos(angle) - (y - center[1]) * np.sin(angle) + center[0]
        rotated_y = (x - center[0]) * np.sin(angle) + (y - center[1]) * np.cos(angle) + center[1]
        rotated_z = z  # No rotation in the z-axis
        rotated_x, rotated_y, rotated_z = rotated_x.astype(np.float32), rotated_y.astype(np.float32), rotated_z.astype(
            np.float32)

        # Calculate the Gaussian distribution with different standard deviations for each axis
        map = np.exp(-((rotated_x - center[0]) ** 2 / (2 * std_x ** 2) +
                       (rotated_y - center[1]) ** 2 / (2 * std_y ** 2) +
                       (rotated_z - center[2]) ** 2 / (2 * std_z ** 2)))

        # Add noise
        noise = np.random.normal(0, map.max() / 50, map.shape).astype(np.float32)
        map = map + noise

        map = normalize_01(map)
        # map = map * np.random.uniform(1, 2) + np.random.uniform(0.1, 0.5)
        map = map + np.random.uniform(0.1, 0.5)

        target = data[..., slb] / map
        # plt.imshow(map[:, :, 10], cmap='gray', origin='lower')
        # plt.show()
        # plt.imshow(np.max(data[...,slb], axis=2), cmap='gray', origin='lower')
        # plt.show()
        # plt.imshow(np.max(target, axis=2), cmap='gray', origin='lower')
        # plt.show()
        # plt.imshow(np.max(target, axis=1), cmap='gray', origin='lower')
        # plt.show()

        targets.append(target.astype(np.float32))

    targets = np.stack(targets, axis=-1)

    np.random.set_state(prev_state)

    return targets


def save_into_slabs(img_path, slb_dir, n_slabs=5):
    img_path, slb_dir = Path(img_path), Path(slb_dir)

    # Create the slab directory if it doesn't exist
    os.makedirs(slb_dir, exist_ok=True)

    # Load the 3D MRA image
    img = nib.load(img_path)
    # x: readout - AP, y: PE - LR
    data = np.asarray(img.dataobj, dtype=np.int16).swapaxes(0, 1)

    # Determine the slab size along the z dimension
    slab_size = data.shape[-1] // n_slabs

    # Split the data into slabs along the z dimension
    slabs = []
    for i in range(n_slabs):
        start = i * slab_size
        # end = (i + 1) * slab_size if i < n_slabs - 1 else data.shape[-1]
        # the last slab may discard a few layers from the top
        end = (i + 1) * slab_size
        slab_data = data[..., start:end]
        slabs.append(slab_data)

    # Save as one npy file
    slabs = np.stack(slabs, axis=-1)
    slabs_filename = f"{img_path.stem.split('.')[0]}_slabs.npy"
    slabs_path = slb_dir / slabs_filename
    np.save(slabs_path, slabs)
    print(f"Slabs of {img_path.name} saved in {slabs_path}")

    # plt.imshow(slabs[:, :, 10, 1], cmap='gray', origin='lower')
    # plt.show()

    return slabs


def read_into_slabs(img_path, n_slabs=5):
    img_path = Path(img_path)

    # Load the 3D MRA image
    img = nib.load(img_path)
    # x: readout - AP, y: PE - LR
    data = np.asarray(img.dataobj, dtype=np.int16).swapaxes(0, 1)

    # Determine the slab size along the z dimension
    slab_size = data.shape[-1] // n_slabs

    # Split the data into slabs along the z dimension
    slabs = []
    for i in range(n_slabs):
        start = i * slab_size
        # end = (i + 1) * slab_size if i < n_slabs - 1 else data.shape[-1]
        # the last slab may discard a few layers from the top
        end = (i + 1) * slab_size
        slab_data = data[..., start:end]
        slabs.append(slab_data)
    slabs = np.stack(slabs, axis=-1)
    # plt.imshow(slabs[:, :, 10, 1], cmap='gray', origin='lower')
    # plt.show()

    return slabs

def read_into_slabs_overlap(img_path, n_slabs=5, slab_size=24, overlap=0.2):
    img_path = Path(img_path)

    # Load the 3D MRA image
    img = nib.load(img_path)
    data = np.asarray(img.dataobj, dtype=np.int16).swapaxes(0, 1)

    # Calculate the overlap in slices
    overlap_slices = int(slab_size * overlap)

    # Split the data into slabs with specified size and overlap
    slabs = []
    for i in range(n_slabs):
        start = i * (slab_size - overlap_slices)
        end = start + slab_size
        if end > data.shape[-1]:  # Adjust for boundary if needed
            end = data.shape[-1]
            start = end - slab_size
        slab_data = data[..., start:end]
        slabs.append(slab_data)

    slabs = np.stack(slabs, axis=-1)
    return slabs