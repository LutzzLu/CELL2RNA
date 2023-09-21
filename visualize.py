import torch
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir("/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/Lutz/CELL2RNA/spatial_omics/code/MaskRCNN")
from utils_1 import getModel
from dataset import ImageDataset

CLASSES = ["Neutrophil", "Epithelial", "Lymphocyte", "Plasma", "Eosinohil", "Connective"]

def visualize(img, pred, label=None, display=False, filename=None):
  plt.figure(figsize=(3,3))
  num_plots = 3 if label is not None else 2
  masks, boxes, pred_cls = pred
  plt.subplot(1, num_plots, 1)
  plt.imshow(img)

  class_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
  for i in range(len(masks)):
    class_map[masks[i]] = pred_cls[i]+1

  plt.subplot(1, num_plots, 2)
  plt.imshow(class_map)
  if label is not None:
    plt.subplot(1, num_plots, 3)
    plt.imshow(label[..., 1])
  if (display):
    plt.show()
  if (filename is not None):
    plt.savefig(f'./output_imgs/{filename}.png', dpi=300)
    plt.close()

def select(arr, idxs):
  return [arr[i] for i in idxs]

def extract_data(pred, threshold=0.5):
  # threshold = 0.5
  pred_score = pred['scores'].detach().cpu().numpy()
  pred_t  = pred_score > threshold
  pred_masks = (pred['masks'] > threshold).squeeze().detach().cpu().numpy()
  pred_class = pred['labels'].cpu().numpy()
  pred_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in list(pred['boxes'].detach().cpu().numpy())]
  pred_boxes = select(pred_boxes, pred_t)
  pred_masks = pred_masks[pred_t]
  pred_class = pred_class[pred_t]
  return pred_masks, pred_boxes, pred_class

def test_model(modelPath):
  # device = torch.device('cpu')
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dataset = ImageDataset()
  imgs = np.load('./data/images.npy')
  model = getModel().to(device)
  state_dict = torch.load(modelPath, map_location=device)
  model.load_state_dict(state_dict)
  model.eval()


  index = int(input("Enter index of image, -1 to quit: "))
  while (index != -1):
    img_tensor = dataset.images[index]
    img_tensor = torch.tensor(np.transpose(img_tensor, axes=[2, 0, 1])).to(device)
    img_tensor = [img_tensor.to(device)]
    pred = model(img_tensor)
    # only one image
    pred = extract_data(pred[0])
    visualize(imgs[dataset.index_map[index]], pred, label=dataset.labels[index], filename=f'interactive_{index}')
    index = int(input("Enter index of image, -1 to quit: "))

def test_boxes():
  dataset = ImageDataset()
  index = 0
  imgs = np.load('./data/images.npy')
  img = imgs[index]
  img = img.astype(np.uint8)
  label = dataset.labels[index]
  boxes, masks, pred_cls = [], [], []
  for i in range(len(dataset.boxes[index])):
    # while there is still boxes left
    if (dataset.boxLabels[index][i][0] != 0 and dataset.boxLabels[index][i][1] != 0):
        box = dataset.boxes[index][i]
        boxes.append(((int(box[0]), int(box[1])), (int(box[2]), int(box[3]))))
        pred_cls.append(np.array(dataset.boxLabels[index][i][1] - 1, dtype=np.int64))
        mask = np.array(label[..., 0] == dataset.boxLabels[index][i][0], dtype=np.uint8)
        masks.append(mask)
    else:
        break
  visualize(img, (masks, boxes, pred_cls), display=True)

if __name__ == "__main__":
  # test_boxes()
  test_model("./models/maskrcnn_resnet50_1_10.pt")