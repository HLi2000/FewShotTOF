from typing import Any, Dict, Optional, Tuple, Union
from functools import partial
from typing import Callable, List
from torchvision.transforms import transforms, InterpolationMode

import numpy as np
import torch

def mask_center(x: torch.Tensor, mask_from: int, mask_to: int) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    mask = torch.zeros_like(x)
    mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]

    return mask


def batched_mask_center(
    x: torch.Tensor, mask_from: torch.Tensor, mask_to: torch.Tensor
) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Can operate with different masks for each batch element.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    if not mask_from.shape == mask_to.shape:
        raise ValueError("mask_from and mask_to must match shapes.")
    if not mask_from.ndim == 1:
        raise ValueError("mask_from and mask_to must have 1 dimension.")
    if not mask_from.shape[0] == 1:
        if (not x.shape[0] == mask_from.shape[0]) or (
            not x.shape[0] == mask_to.shape[0]
        ):
            raise ValueError("mask_from and mask_to must have batch_size length.")

    if mask_from.shape[0] == 1:
        mask = mask_center(x, int(mask_from), int(mask_to))
    else:
        mask = torch.zeros_like(x)
        for i, (start, end) in enumerate(zip(mask_from, mask_to)):
            mask[i, :, :, start:end] = x[i, :, :, start:end]

    return mask

def resize_target(tensor, target_shape):
    """
    Resize the input tensor to match the target shape if its dimensions are larger.

    Args:
        tensor (torch.Tensor): Input tensor to be resized.
        target_shape (tuple): Target shape

    Returns:
        torch.Tensor: Resized tensor.
    """
    # Get the minimum shape
    current_shape = list(tensor.shape[1:3])
    target_shape = [min(target_shape[idx], current_shape[idx]) for idx in range(len(target_shape))]

    if target_shape != current_shape:
        tensor = torch.movedim(tensor, source=[1, 2], destination=[2, 3])
        resizer = transforms.Resize(target_shape, InterpolationMode.NEAREST)
        tensor = resizer(tensor)
        tensor = torch.movedim(tensor, source=[2, 3], destination=[1, 2])

    return tensor

def crop_or_zero_pad(tensor, target_shape):
    """
    Crop or zero-pad the input tensor to match the target shape.

    Args:
        tensor (torch.Tensor): Input tensor to be cropped or zero-padded.
        target_shape (tuple): Target shape in the format (depth, height, width).

    Returns:
        torch.Tensor: Cropped or zero-padded tensor.
    """
    # Get the current shape of the tensor
    current_shape = tensor.shape

    # Initialize cropping and padding parameters
    crop_params = [0] * len(current_shape)
    pad_params = [(0, 0)] * len(current_shape)

    # Determine cropping and padding parameters for each dimension
    for dim in range(len(current_shape)):
        if dim == 0:
            crop_params[dim] = slice(0, current_shape[dim])
            continue
        if current_shape[dim] > target_shape[dim]:
            # Calculate cropping parameters
            crop_start = (current_shape[dim] - target_shape[dim]) // 2
            crop_end = crop_start + target_shape[dim]
            crop_params[dim] = slice(crop_start, crop_end)
        elif current_shape[dim] <= target_shape[dim]:
            crop_params[dim] = slice(0, current_shape[dim])
            # Calculate padding parameters
            pad_width = target_shape[dim] - current_shape[dim]
            pad_params[dim] = (pad_width // 2, pad_width - pad_width // 2)

    # Crop or zero-pad the tensor
    cropped_or_padded_tensor = torch.nn.functional.pad(tensor[crop_params], sum(pad_params[::-1], ()))

    return cropped_or_padded_tensor

def normalize_1_kspace(
    kspace: torch.Tensor,
    img: torch.Tensor,
    eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Normalize the given tensor to [0, 1]

    Applies the formula (data - min) / (max - min + eps).

    Args:
        data: Input data to be normalized.
        eps: Added to prevent dividing by zero.

    Returns:
        Normalized tensor.
    """
    return kspace / (img.max() + eps)

def normalize_1(
    data: torch.Tensor,
    eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Normalize the given tensor to [0, 1]

    Applies the formula (data - min) / (max - min + eps).

    Args:
        data: Input data to be normalized.
        eps: Added to prevent dividing by zero.

    Returns:
        Normalized tensor.
    """
    return data / (data.max() + eps)

def normalize_01_kspace(
    kspace: torch.Tensor,
    img: torch.Tensor,
    eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Normalize the given tensor to [0, 1]

    Applies the formula (data - min) / (max - min + eps).

    Args:
        data: Input data to be normalized.
        eps: Added to prevent dividing by zero.

    Returns:
        Normalized tensor.
    """
    # scalar multiplication
    kspace = kspace / (img.max() - img.min() + eps)

    # addition
    x, y, z = kspace.shape[1:4]
    kspace[:, x//2, y//2, z//2, :] += (- img.min() / (img.max() - img.min() + eps))

    return kspace

def normalize_01(
    data: torch.Tensor,
    eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Normalize the given tensor to [0, 1]

    Applies the formula (data - min) / (max - min + eps).

    Args:
        data: Input data to be normalized.
        eps: Added to prevent dividing by zero.

    Returns:
        Normalized tensor.
    """
    return (data - data.min()) / (data.max() - data.min() + eps)

def normalize(
    data: torch.Tensor,
    mean: Union[float, torch.Tensor],
    stddev: Union[float, torch.Tensor],
    eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Normalize the given tensor.

    Applies the formula (data - mean) / (stddev + eps).

    Args:
        data: Input data to be normalized.
        mean: Mean value.
        stddev: Standard deviation.
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        Normalized tensor.
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(
    data: torch.Tensor, eps: Union[float, torch.Tensor] = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize the given tensor  with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.

    Args:
        data: Input data to be normalized
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps)


class Repr:
    """Evaluatable string representation of an object"""

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"


class FunctionWrapperSingle(Repr):
    """A function wrapper that returns a partial for input only."""

    def __init__(self, function: Callable, *args, **kwargs):
        self.function = partial(function, *args, **kwargs)

    def __call__(self, inp: np.ndarray):
        return self.function(inp)


class FunctionWrapperDouble(Repr):
    """A function wrapper that returns a partial for an input-target pair."""

    def __init__(
        self,
        function: Callable,
        input: bool = True,
        target: bool = False,
        *args,
        **kwargs,
    ):
        self.function = partial(function, *args, **kwargs)
        self.input = input
        self.target = target

    def __call__(self, inp, tar):
        if self.input:
            inp = self.function(inp)
        if self.target:
            tar = self.function(tar)
        return inp, tar


class Compose:
    """Baseclass - composes several transforms together."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __repr__(self):
        return str([transform for transform in self.transforms])


class ComposeDouble(Compose):
    """Composes transforms for input-target pairs."""

    def __call__(self, inp: np.ndarray, target: dict):
        for t in self.transforms:
            inp, target = t(inp, target)
        return inp, target


class ComposeSingle(Compose):
    """Composes transforms for input only."""

    def __call__(self, inp: np.ndarray):
        for t in self.transforms:
            inp = t(inp)
        return inp
