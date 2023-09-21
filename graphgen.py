import numpy as np
import pandas as pd
import PIL
class neighborFinder():
    def __init__(self, xy, tile_factor=100):
        self.xy = xy
        self.tf = tile_factor
        self.build()
    def build(self):
        self.posdict = {}
        for i in range(len(self.xy)):
            x, y = self.xy[i]
            key = (x // self.tf, y // self.tf)
            if key not in self.posdict:
                self.posdict[key] = [i]
            else:
                self.posdict[key].append(i)
    def find(self, pos, rad):
        x, y = pos
        nodes = []
        for sx in range((x - rad) // self.tf, 1 + (x + rad) // self.tf):
            for sy in range((y - rad) // self.tf, 1 + (y + rad) // self.tf):
                key = (sx, sy)
                if (key not in self.posdict): continue;
                indexes = self.posdict[key]
                for index in indexes:
                    dx = self.xy[index][0] - x; dy = self.xy[index][1] - y;
                    if (dx*dx + dy*dy <= rad*rad):
                        nodes.append(index)
        return nodes

def getCellPatch(x, y, image, size):
    if (y - size < 0 or y + size >= image.shape[0] or x - size < 0 or x + size >= image.shape[1]):
        blank = np.zeros((2*size + 1, 2*size + 1, 3), dtype=np.uint8)
        ly = max(y-size, 0); my = min(y+size, image.shape[0]);
        lx = max(x-size, 0); mx = min(x+size, image.shape[1]);
        blank[:, :] = [int(x) for x in np.mean(image[ly:my, lx:mx, :], axis=(0,1))]
        blank[ly-(y-size):my-(y-size), lx-(x-size):mx-(x-size), :] = image[ly:my,lx:mx :]
        patch = blank
    else:
        patch = image[y-size:y+size+1, x-size:x+size+1, :]
    return patch

def _generateCellPatches(name, data):
    size = 50
    cellpatches = []
    image = np.array(PIL.Image.open(data.datadir + '/raw_data/' + name + '.TIF'))
    for index in range(len(data.detxy[name])):
        x, y = data.detxy[name][index]
        cellpatches.append(getCellPatch(x,y,image,size))
    return cellpatches

def _generateCellLabels(name, data):
    n = neighborFinder(data.nodexy[name])
    # try smaller radius... 500 -> 125 -> 75
    rad = 75
    indexes, labels = [], []
    avg_neighbors = 0
    total_zero = 0
    eps = 1e-5
    for i, pos in enumerate(data.detxy[name]):
        x, y = pos
        totalweight = 0
        nemb = np.zeros((1,1000))
        neighbors = n.find((x,y), rad)
        avg_neighbors += len(neighbors)
        for index in neighbors:
            nx, ny = data.nodexy[name][index]
            emb = data.gene_ex[name][index]
            dist = np.sqrt((x-nx)**2 + (y-ny)**2) + eps
            totalweight += 1 / (dist**2)
            nemb += emb / (dist**2)
            
        if (totalweight != 0): 
            nemb /= totalweight
            indexes.append(i)
        else: 
            total_zero += 1
            
        labels.append(nemb)

    print(f"Average number of visium nodes used per embedding: {avg_neighbors/len(data.detxy[name])}")
    print(f"Total zero embeddings: {total_zero}")
    return labels, indexes

def generateCellPatches(snames, data):
    cellpatches = {}
    for name in snames:
        cellpatches[name] = _generateCellPatches(name, data)
    return cellpatches

def generateCellLabels(snames, data):
    labels = {}
    indexes = {}
    for name in snames:
        labels[name], indexes[name] = _generateCellLabels(name, data)
    return labels, indexes