import numpy as np
import torch
def numpify(a):
    if isinstance(a, np.ndarray):
        return a
    return a.detach().cpu().numpy()

def select(arr, ind):
    newarr = []
    for i in ind:
        newarr.append(arr[i])
    return newarr