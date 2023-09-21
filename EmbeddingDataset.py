import os
import torch
from graphgen import neighborFinder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
import random

class EmbeddingDataset(Dataset):
    def __init__(self, patches, labels, image_transform=None, target_transform=None):
        self.patches = patches
        self.labels = labels
        self.image_transform = image_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.patches)
    
    def select(a, idxs):
        subset = []
        for idx in idxs:
            subset.append(a[idx])
        return subset

    def __getitem__(self, idx):
        patch = self.patches[idx]
        #print(f'{patch.shape=}')
        if self.image_transform:
            patch = self.image_transform(patch)
        if self.labels is None:
            # dummy value that will throw an error if you try to use it
            return patch.float(), torch.tensor([0])
        else:
            label = self.labels[idx]
            if self.target_transform:
                label = self.target_transform(label)
            return patch.float(), label.float()

def train_val_dataset(dataset, detxy, size, val_split=0.10):
    # ensure val and train have different sets of connected components.
    off = 0
    train_idx, val_idx = [], []
    for slide in detxy:
        cc = connectedComponents(detxy[slide], size).getComponents()
        random.shuffle(cc)
        num_val = len(detxy[slide]) * val_split 
        split_index = 0; num_split = 0
        while (num_split < num_val):
            num_split += len(cc[split_index])
            split_index += 1
        for c in cc[split_index:]:
            train_idx.extend([e+off for e in c])
        for c in cc[0:split_index]:
            val_idx.extend([e+off for e in c])
        off += len(detxy[slide])
    # valid shuffle.    
    print(f"Valid Shuffle: {len(train_idx) == len(set(train_idx))}")
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

class connectedComponents():
    def __init__(self, positions, size):
        self.num_connected = 0
        self.n = neighborFinder(positions)
        self.visited = [0] * len(positions)
        self.components = [0] * len(positions)
        self.positions = positions
        self.size = size
    
    def getComponents(self):
        self.floodfill()
        components = [[] for i in range(self.num_connected)]
        for i in range(len(self.positions)):
            components[self.components[i]].append(i)
        return components

    def floodfill(self):
        for i in range(len(self.positions)):
            if (self.visited[i] == 0):
                self.visited[i] = 1
                self.components[i] = self.num_connected
                self.dfs(i)
                self.num_connected += 1

    def dfs(self, index):
        pos = self.positions[index]
        for index in self.n.find(pos, self.size):
            if (self.visited[index] == 1): continue;
            self.visited[index] = 1
            self.components[index] = self.num_connected
            self.dfs(index)