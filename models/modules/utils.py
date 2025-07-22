import torch
from typing import Tuple, Sequence

def resize_and_normalize_tensor(
        batch_images: torch.Tensor,
        size: Tuple[int, int],
        mean: Sequence[float],
        std: Sequence[float]
) -> torch.Tensor:
    """
    Resize and normalize a batch of images.

    Parameters
    ----------
        batch_images (torch.Tensor): A tensor of shape (B, C, H, W) representing a batch of images.
        size (Tuple[int, int]): Target spatial size (height, width) for resizing.
        mean (Sequence[float]): Per-channel mean for normalization (length must be C).
        std (Sequence[float]): Per-channel standard deviation for normalization (length must be C).

    Returns
    -------
        torch.Tensor: A tensor of shape (B, C, H_out, W_out) with resized and normalized images.
    """
    resized_tensor = torch.nn.functional.interpolate(batch_images, size=size, mode='bilinear', align_corners=False)
    mean_tensor = torch.tensor(mean, device=batch_images.device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=batch_images.device).view(1, 3, 1, 1)
    normalized = (resized_tensor - mean_tensor) / std_tensor
    return normalized