from typing import List

import torch
import torch.utils.data

from .patchesdataset import PatchesDataset, create_patches_dataset


class SlidesDataset(torch.utils.data.Dataset):
    """
    A SlidesDataset is a wrapper for a list of PatchesDatasets that are separated based on the slide they are associated with.
    Note: All slides must have the same number of genes.
    """

    def __init__(self, slides: List[PatchesDataset], transform=None):
        super().__init__()

        self.slides = slides
        self.n_genes = slides[0].n_genes
        self.transform = transform

        for slide in slides:
            assert slide.n_genes == self.n_genes, "All slides must have the same number of genes"

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, index: int):
        slide = self.slides[index]
        if self.transform is not None:
            slide = self.transform(slide)

        return slide

    def merge(self):
        return torch.utils.data.ConcatDataset([slide for slide in self.slides])


def create_slides_dataset(training_folds, patch_size=512, binary=True, slide_transform=None, patch_transform=None, gene_subset=None):
    """
    Creates a SlidesDataset based on several SpatialOmicsSlide objects or strings representing the ID of the
    SpatialOmicsSlide to load. An optional transform may be provided, which is passed to each PatchesDataset.
    """

    return SlidesDataset(
        [create_patches_dataset(
            training_fold=training_folds[i],
            patch_size=patch_size,
            binary=binary,
            transform=patch_transform,
            gene_subset=gene_subset,
        ) for i in range(len(training_folds))],
        slide_transform,
    )
