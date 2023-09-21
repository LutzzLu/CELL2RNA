import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib
from matplotlib import pyplot as plt
import os

def plotAUROC(snames, labels: np.array, preds: np.array, med_ex: np.array, binary=False):
    fig, axs = plt.subplots(len(snames), 1)
    for i, name in enumerate(snames):
        num_genes = 1000
        if not binary:
            A = (labels[name] > med_ex[name]).astype(int).reshape(-1, num_genes)
            P = (preds[name] > med_ex[name]).astype(int).reshape(-1, num_genes)
        else:
            A = labels[name].astype(float).reshape(-1, num_genes)
            P = preds[name].astype(float).reshape(-1, num_genes)
        roc_auc_scores = np.zeros(num_genes)
        count = 0
        difference = 0
        for gene in range(num_genes):
            if (np.any(A[:, gene] == 0) and np.any(A[:, gene] == 1)):
                # Has both classes
                difference += abs(np.mean(A[:, gene] - P[:, gene]))
                roc_auc_scores[gene] = roc_auc_score(A[:, gene], P[:, gene])
                count += 1
            else:
                # Put -1 if ROC curve is not defined
                roc_auc_scores[gene] = -1
        axs[i].hist(roc_auc_scores[roc_auc_scores >= 0], bins=20)
        print(f"Median AUROC - {np.median(roc_auc_scores[roc_auc_scores >= 0])}")

def plotAP(snames, labels: np.array, preds: np.array, med_ex: np.array, binary=False):
    fig, axs = plt.subplots(len(snames), 1)
    for i, name in enumerate(snames):
        num_genes = 1000
        if not binary:
            A = (labels[name] > med_ex[name]).astype(int).reshape(-1, num_genes)
            P = (preds[name] > med_ex[name]).astype(int).reshape(-1, num_genes)
        else:
            A = labels[name].astype(float).reshape(-1, num_genes)
            P = preds[name].astype(float).reshape(-1, num_genes)

        ap_scores = np.zeros(num_genes)
        for gene in range(num_genes):
            if (np.any(A[:, gene] == 0) and np.any(A[:, gene] == 1)):
                ap_scores[gene] = average_precision_score(A[:, gene], P[:, gene])
            else:
                ap_scores[gene] = -1
        axs[i].hist(ap_scores[ap_scores >= 0], bins=20)
        print(f"Median AP - {np.median(ap_scores[ap_scores >= 0])}")

def plotHeatmaps(image, index, labels: np.array, xys: np.array, binary=False):
    cmap = matplotlib.cm.get_cmap('viridis')
    if binary:
        intensities = labels[:, index]
    else:
        intensities = 1 / (1 + np.exp(-labels[:, index]));
    colors = cmap(intensities, alpha=0.5);
    heatmap = draw_heatmap(np.copy(image), 5, xys, colors, 100);
    return heatmap


def draw_heatmap(image: np.ndarray, downsample: int, xys: np.ndarray, colors: np.array, size: int):
    # Generate "true" photo
    new_image: np.ndarray = image[::downsample, ::downsample, :].copy()
    size = size // downsample
    for i, (xy, color) in enumerate(zip(xys, colors)):
        x, y = xy
        # Center around gene locations
        x = int(x) // downsample - size // 2
        y = int(y) // downsample - size // 2

        if len(color) == 3:
            alpha = 0.50
        else:
            alpha = color[3]
        new_image[y:y + size, x:x + size] = np.uint8((1 - alpha) * new_image[y:y + size, x:x + size].astype(np.float32) + 255 * alpha * color[:3])
    return new_image