import numpy as np
import tqdm
import torch
from torchvision.models.detection.mask_rcnn import MaskRCNN, MaskRCNNPredictor, MaskRCNNHeads, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.resnet import resnet50
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FastRCNNConvFCHead, RPNHead, _default_anchorgen
from torch import nn

def test():
    print("Bye!")

def generate_bounding_boxes(labels):
    boxes = []
    boxLabels = []
    for index in tqdm.tqdm(range(len(labels))):
        boxes.append([])
        boxLabels.append([])
        label = labels[index]
        instToClass = [-1] * 1000
        minx = [256] * 1000
        miny = [256] * 1000
        maxx = [-1] * 1000
        maxy = [-1] * 1000
        for i in range(label[..., 0].shape[0]):
            for j in range(label[..., 0].shape[1]):
                value = label[i, j, 0]
                if value != 0:
                    if minx[value] > j:
                        minx[value] = j
                    if miny[value] > i:
                        miny[value] = i
                    if maxx[value] < j:
                        maxx[value] = j
                    if maxy[value] < i:
                        maxy[value] = i
                    instToClass[value] = label[i,j,1]

        for i in range(1, len(minx)):
            if (maxx[i] - minx[i]) > 0 and (maxy[i] - miny[i]) > 0:
                boxes[index].append((minx[i], miny[i], maxx[i], maxy[i]))
                boxLabels[index].append((i, instToClass[i]))
    return boxes, boxLabels

def save3DArray(boxes, filename=None):
    # save list of lists of tuples
    array = np.zeros((len(boxes), 1000, 4), dtype=np.uint8)
    for i in range(len(boxes)):
        for j in range(len(boxes[i])):
            for k in range(len(boxes[i][j])):
                array[i, j, k] = boxes[i][j][k]
    np.save(filename, array)

def sel(arr, idxs):
    return [arr[i] for i in range(len(idxs)) if idxs[i]]

def extractData(pred, threshold=0.5, debug=False):
    pred_score = pred['scores'].detach().cpu().numpy().reshape(-1)
    pred_t = pred_score > threshold
    pred_masks = (pred['masks'] > threshold).squeeze().detach().cpu().numpy().reshape(-1,256,256)
    pred_class = pred['labels'].detach().cpu().numpy().reshape(-1)
    pred_boxes = [[int(i[0]), int(i[1]), int(i[2]), int(i[3])] for i in list(pred['boxes'].detach().cpu().numpy())]
    if (debug):
        print(len(pred_boxes))
    return pred_masks[pred_t], pred_class[pred_t], sel(pred_boxes, pred_t), pred_score[pred_t]

def getModel(modelPath=None):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    num_classes = 6
    model = customMaskRCNN(weights='MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT', num_classes=num_classes)
    if modelPath is None:
        model = model.to(device)
        return model
    
    modelPath = f'{modelPath}.pt' if modelPath[-3:] != '.pt' else modelPath
    state_dict = torch.load(modelPath, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model

def customMaskRCNN(*, weights = None, num_classes = None, trainable_backbone_layers = None, **kwargs):
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.verify(weights)

    is_trained = True
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)

    backbone = resnet50(weights=None, progress=True)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers, norm_layer=nn.BatchNorm2d)
    rpn_anchor_generator = _default_anchorgen() # AnchorGenerator(sizes=((8, 16, 32, 48),), aspect_ratios=((0.5, 1.0, 2.0),))
    rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], conv_depth=2)
    box_head = FastRCNNConvFCHead(
        (backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d
    )
    mask_head = MaskRCNNHeads(backbone.out_channels, [256, 256, 256, 256], 1, norm_layer=nn.BatchNorm2d)
    model = MaskRCNN(
        backbone,
        num_classes=len(weights.meta["categories"]),
        rpn_anchor_generator=rpn_anchor_generator,
        rpn_head=rpn_head,
        box_head=box_head,
        mask_head=mask_head,
        box_detections_per_img=500,
        **kwargs,
    )
    # load pretrained model [this won't work if you change the model!]
    model.load_state_dict(weights.get_state_dict(progress=True))

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

def main():
    pass

if __name__ == "__main__":
    main()