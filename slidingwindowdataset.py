"""
Dataset for Vision Transformer:
 * Creates a sliding window
"""

from torch.utils.data import Dataset
import torch

class SlidingWindowDataset(Dataset):
    def __init__(self, image: torch.Tensor, patch_size: int, patch_xy: torch.Tensor, window_n_patches: int, labels: torch.Tensor):
        self.image = image
        self.patch_size = patch_size
        self.patch_xy = patch_xy
        self.window_n_patches = window_n_patches
        self.labels = labels
