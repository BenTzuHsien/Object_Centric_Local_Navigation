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
    mean_tensor = torch.tensor(mean).view(1, 3, 1, 1).to(batch_images)
    std_tensor = torch.tensor(std).view(1, 3, 1, 1).to(batch_images)
    normalized = (resized_tensor - mean_tensor) / std_tensor
    return normalized

def get_masked_region(masks):

    B = masks.shape[0]
    nonzero_indices = torch.nonzero(masks)
    b, y, x = nonzero_indices[:, 0], nonzero_indices[:, 2], nonzero_indices[:, 3]

    x1 = torch.zeros(B).to(masks)
    y1 = torch.zeros(B).to(masks)
    x2 = torch.zeros(B).to(masks)
    y2 = torch.zeros(B).to(masks)

    x1.scatter_reduce_(0, b, x.to(masks), reduce='amin', include_self=False)
    y1.scatter_reduce_(0, b, y.to(masks), reduce='amin', include_self=False)
    x2.scatter_reduce_(0, b, x.to(masks), reduce='amax', include_self=False)
    y2.scatter_reduce_(0, b, y.to(masks), reduce='amax', include_self=False)

    return torch.stack([x1, y1, x2, y2], dim=1).int()