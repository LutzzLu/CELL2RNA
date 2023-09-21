#from functools import cached_property
import os
from typing import List, Optional, Union
import numpy as np

import torch
import torchvision.transforms.functional as TF
from device import device


os.chdir("/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/Lutz/CELL2RNA/spatial_omics/code/data/")

from spatialomicsslide import SpatialOmicsSlide
from spatialomicsslide import load_training_data

def preprocess_wsi(data: SpatialOmicsSlide, patch_size: int, binary: bool):
    """
    Data processing. Split into patches of size 512x512.
    Then, for each patch, find whether the expression of
    a given gene is above or below the median expression
    of that gene across the whole slide.
    """

    gene_ex = data.gene_ex.to(device)
    X = data.gene_ex_X
    Y = data.gene_ex_Y
    xy = torch.stack([data.gene_ex_X, data.gene_ex_Y], dim=1)

    # select only valid xys
    valid_indexes = \
        (X >= (patch_size // 2)) \
        & (Y >= (patch_size // 2)) \
        & (X < (data.image.shape[1] - patch_size // 2)) \
        & (Y < (data.image.shape[0] - patch_size // 2))

    gene_ex = gene_ex[valid_indexes]
    xy = xy[valid_indexes]

    if binary:
        median: torch.Tensor = torch.median(gene_ex, dim=0)[0]
        gene_ex_binary = (gene_ex > median).float()
        return xy, gene_ex_binary
    else:
        return xy, gene_ex


class PatchesDataset(torch.utils.data.Dataset):
    """
    A PatchesDataset is a bunch of patches with positional information and corresponding genes.
    This dataset format can be used for several slides in aggregate when training a model that is position-agnostic,
    or it can also be used for individual slides to train models that require positional information.
    To create a dataset of slides that need to be separated from each other, use a SlidesDataset.
    """

    def __init__(
        self,
        image: torch.Tensor, # or np.ndarray[uint8]
        labels: Optional[torch.Tensor],
        patch_xys: torch.Tensor,
        patch_size: int,
        transform=None,
        genes: List[str] = None,
        normalizer=None):
        super().__init__()

        if genes is None:
            raise ValueError("genes must be provided")

        self.image = image
        self.patch_xys = patch_xys
        self.labels = labels
        self.transform = None
        self.patch_size = patch_size
        self.genes = genes
        self.normalizer = normalizer
        if labels is not None:
            self.n_genes = labels.shape[1]
        else:
            self.n_genes = -1

        self.return_locs = False

    def set_patch_size(self, patch_size: int):
        # del self.patches
        self.patch_size = patch_size

    #@cached_property
    def patches(self):
        patches = torch.zeros((self.patch_xys.shape[0], 3, self.patch_size, self.patch_size), dtype=torch.float, device=device, requires_grad=False)
        for i, (x, y) in enumerate(self.patch_xys):
            patches[i] = self._patch(x, y)
        return patches

    def _patch(self, x, y):
        # Using in-place operations here sucks
        x = x - self.patch_size // 2
        y = y - self.patch_size // 2

        if self.normalizer is not None:
            image = self.image[y:y + self.patch_size, x:x + self.patch_size, :]
            assert type(image) == np.ndarray
            image = self.normalizer.transform(image)
            image = TF.to_tensor(image)
        else:
            image = self.image[:, y:y + self.patch_size, x:x + self.patch_size]
        return image

    def __len__(self):
        return self.patch_xys.shape[0]

    def __getitem__(self, index: int):
        x, y = self.patch_xys[index]

        patch = self._patch(x, y).to(device)
        if patch.shape[1] != self.patch_size:
            raise ValueError(f"Patch at {x}, {y} is {patch.shape} but expected {self.patch_size}x{self.patch_size}")

        assert patch.shape == (3, self.patch_size, self.patch_size)
        if self.transform is not None:
            patch = self.transform(patch)

        if self.labels is not None:
            label = self.labels[index].float()
        else:
            label = None
        
        loc = self.patch_xys[index]

        if self.return_locs:
            return patch, label, loc
        else:
            return patch, label


class StandardScaler:
    def __call__(self, image):
        mean = torch.mean(image, dim=(1, 2), keepdim=True)
        std = torch.std(image, dim=(1, 2), keepdim=True)
        return (image - mean) / std


class PatchesDatasetRandomVerticalFlip:
    """
    Performs a random vertical flip of a patches dataset. This does not
    change the order of the internal list of patches; it just changes the Y values
    relative to each other.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample: PatchesDataset):
        if torch.rand(()) >= self.p:
            return sample

        max_y = sample.image.shape[1]
        new_xys = sample.patch_xys.clone()
        new_xys[:, 1] = max_y - new_xys[:, 1] - sample.patch_size
        new_image = sample.image.flip(1)
        return PatchesDataset(
            new_image,
            sample.labels,
            new_xys,
            sample.patch_size,
            sample.transform
        )


class PatchesDatasetRandomHorizontalFlip:
    """
    Performs a random horizontal flip of a patches dataset. This does not
    change the order of the internal list of patches; it just changes the X values
    relative to each other.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample: PatchesDataset):
        if torch.rand(()) >= self.p:
            return sample

        max_x = sample.image.shape[2]
        new_xys = sample.patch_xys.clone()
        new_xys[:, 0] = max_x - new_xys[:, 0] - sample.patch_size
        new_image = sample.image.flip(2)
        return PatchesDataset(
            new_image,
            sample.labels,
            new_xys,
            sample.patch_size,
            sample.transform,
        )


def create_patches_dataset(
    training_fold: Union[str, SpatialOmicsSlide],
    patch_size=512,
    binary=True,
    transform=None,
    gene_subset=None
):
    if type(training_fold) is str:
        dataset = load_training_data(training_fold)
    else:
        dataset: SpatialOmicsSlide = training_fold

    if dataset.gene_ex is not None:
        patch_xys, labels = preprocess_wsi(dataset, patch_size, binary)

        if gene_subset is not None:
            subset_indexes = [dataset.genes.index(
                gene) for gene in gene_subset if gene in dataset.genes]
            labels = labels[:, subset_indexes]
            genes = [gene for gene in gene_subset if gene in dataset.genes]
        else:
            genes = dataset.genes

    else:
        raise NotImplementedError("Unlabeled slides are not accounted for yet")

    image_pt = torch.from_numpy(dataset.image).permute(2, 0, 1) / 255.0

    return PatchesDataset(image_pt, labels, patch_xys, patch_size, transform, genes)


def load_unlabeled_slide(image, patch_size=768, patch_stride=None, transform=None, genes=None, normalizer=None, is_valid_patch=None) -> PatchesDataset:
    """
    Creates a PatchesDataset from a whole slide image.
    Parameters:
    - image: A torch.Tensor in (C, H, W) format
    - patch_size: The size of the patches to extract (passed to the PatchesDataset)
    - patch_stride: The stride between patches. If None, defaults to patch_size
    - transform: A torchvision transform to apply to each patch (passed to the PatchesDataset)
    - genes: The list of possible genes that can be output (passed to the PatchesDataset)
    - normalizer: The stain normalizer (passed to the PatchesDataset)
    - is_valid_patch: A function that takes a torch.Tensor in (C, H, W) format and returns a boolean
    """

    assert type(image) == torch.Tensor and image.shape[0] in [1, 3], "Image must be a tensor in (C, H, W) format"

    if patch_stride is None:
        patch_stride = patch_size

    tiles_width = (image.shape[-1] - patch_size) // patch_stride
    tiles_height = (image.shape[-2] - patch_size) // patch_stride

    xy = torch.zeros((tiles_height, tiles_width, 2), device=device, dtype=torch.long)
    for i in range(1, tiles_width + 1):
        xy[:, i - 1, 0] = i * patch_stride + patch_size // 2
    
    for j in range(1, tiles_height + 1):
        xy[j - 1, :, 1] = j * patch_stride + patch_size // 2
        
    xy = xy.reshape(-1, 2)

    # Remote invalid patches according to is_valid_patch(patch) function
    valid_patches = torch.zeros((xy.shape[0],), dtype=torch.bool, device=device)
    for i in range(xy.shape[0]):
        x, y = xy[i]
        patch = image[:, y - patch_size // 2:y + patch_size // 2, x - patch_size // 2:x + patch_size // 2]
        valid_patches[i] = is_valid_patch(patch)
    
    xy = xy[valid_patches]

    patches_dataset = PatchesDataset(
        image,
        None,
        xy,
        patch_size,
        transform,
        genes=genes,
        normalizer=normalizer
    )

    return patches_dataset
