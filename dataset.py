import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import copy

class ImageDataset():
    def __init__(self, imgPath="./data/images.npy", labelPath="./data/labels.npy", boxPath="./data/boxes.npy", boxLabelPath="./data/boxLabels.npy", countsPath="./data/counts.csv", range=None):
        # self.device = torch.device("cpu")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.count = pd.read_csv(countsPath)
        self.range = len(self.count) if range is None else range
        self.images = preprocess(np.load(imgPath).astype('float32'))[0:range]
        self.labels = np.load(labelPath).astype('float32')[0:range]
        self.boxes = np.load(boxPath).astype('float32')[0:range]
        self.boxLabels = np.load(boxLabelPath).astype('float32')[0:range]
        self.prune()
        self.train_test_split()

    def prune(self):
        # prune data to only include images with more than 1 label
        self.index_map = []
        counts = self.count.sum(axis=1)
        new_images, new_labels, new_boxes, new_boxesLabels = [], [], [], []
        for i in range(len(self.images)):
            if counts[i] > 0:
                new_images.append(self.images[i])
                new_labels.append(self.labels[i])
                new_boxes.append(self.boxes[i])
                new_boxesLabels.append(self.boxLabels[i])
                self.index_map.append(i)
        self.images = np.array(new_images)
        self.labels = np.array(new_labels)
        self.boxes = np.array(new_boxes)
        self.boxLabels = np.array(new_boxesLabels)
    
    def train_test_split(self):
        indexes = list(range(len(self.images)))
        train_idx, val_idx = train_test_split(indexes, test_size=0.2, random_state=42)
        train_img_data, train_box_data, train_boxLabel_data, train_labels = [], [], [], []
        val_img_data, val_box_data, val_boxLabel_data, val_labels = [], [], [], []
        for i in range(len(train_idx)):
            train_img_data.append(self.images[train_idx[i]])
            train_box_data.append(self.boxes[train_idx[i]])
            train_boxLabel_data.append(self.boxLabels[train_idx[i]])
            train_labels.append(self.labels[train_idx[i]])
        for i in range(len(val_idx)):
            val_img_data.append(self.images[val_idx[i]])
            val_box_data.append(self.boxes[val_idx[i]])
            val_boxLabel_data.append(self.boxLabels[val_idx[i]])
            val_labels.append(self.labels[val_idx[i]])
        self.train_data = (train_img_data, train_box_data, train_boxLabel_data, train_labels)
        self.val_data = (val_img_data, val_box_data, val_boxLabel_data, val_labels)
    
    def get_batches(self, batch_size, training=True):
        data = self.train_data if training else self.val_data
        indexes = list(range(len(data[0])))
        perm_idx = np.random.permutation(indexes) if training else indexes
        for i in range(len(data[0]) // batch_size):
            batch_idx = perm_idx[i * batch_size : (i + 1) * batch_size]
            img_batch, target_batch = [], []
            for j in batch_idx:
                img = copy.deepcopy(data[0][j])
                img = torch.tensor(np.transpose(img, axes=[2, 0, 1])).to(self.device)
                img = img.to(self.device)
                img_batch.append(img)
                target_batch.append(self.make_target(data[1][j], data[2][j], data[3][j]))
            batch = (img_batch, target_batch)
            yield batch
    
    def make_target(self, boxes, boxLabels, label):
        data = {}
        img_boxes = []
        img_class_labels = []
        img_masks = []
        for i in range(len(boxes)):
            # while there is still boxes left
            if (boxLabels[i][0] != 0 and boxLabels[i][1] != 0):
                invalid_box = False
                for j in range(4):
                    if boxes[i][j] < 0 or boxes[i][j] >= 255:
                        invalid_box = True
                if invalid_box:
                    continue
                img_boxes.append(boxes[i])
                # classes are 0-indexed.
                img_class_labels.append(np.int64(boxLabels[i][1] - 1))
                img_mask = np.array(label[..., 0] == boxLabels[i][0], dtype=np.uint8)
                img_masks.append(img_mask)
            else:
                break
        data['boxes'] = torch.tensor(np.stack(img_boxes)).to(self.device)
        data['labels'] = torch.tensor(np.stack(img_class_labels)).to(self.device)
        data['masks'] = torch.tensor(np.stack(img_masks)).to(self.device)
        return data

def preprocess(images):
    # preprocess images
    images = images / 255   
    images = np.transpose(images, axes=[3, 0, 1, 2])
    # normalize each channel
    for i in range(len(images)):
        mean = np.mean(images[i])
        std = np.std(images[i])
        images[i] = (images[i] - mean) / std
    images = np.transpose(images, axes=[1, 2, 3, 0])
    return images