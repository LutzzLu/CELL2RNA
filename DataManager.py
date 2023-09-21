from spatialomicsslide import SpatialOmicsSlide, load_training_data
from patchesdataset import preprocess_wsi
import PIL
import pickle
import numpy as np
import torch
from utils import numpify

DATA_DIR = '/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/projects/spatial_omics/DH/visium'
class dataManager():
    def __init__(self, snames, datadir=DATA_DIR, load_patches=False, load_detections=True, load_images=True):
        self.snames = snames
        self.datadir = datadir
        self.images = {}
        self.patches = {}
        self.med_ex = {}
        self.gene_ex = {}
        self.gene_ex_binary = {}
        self.patchxy = {}
        self.nodexy = {}
        if load_patches:
            self.fetchPatches()
        if load_detections:
            self.load_detections()
        if load_images:
            self.loadImages()
        self.fetchExp()
    
    def loadImages(self):
        for name in self.snames:
            self.images[name] = np.array(PIL.Image.open(self.datadir + '/raw_data/' + name + '.TIF'))
        
    def fetchPatches(self):
        psize = 256
        print("Generating Patches...")
        for name in self.snames:
            image = np.array(PIL.Image.open(self.datadir + '/raw_data/' + name + '.TIF'))
            ps, xys = [], []
            for i in range(0, image.shape[0]-psize, psize):
                for j in range(0, image.shape[1]-psize, psize):
                    ps.append(image[i:i+psize, j:j+psize, :])
                    xys.append([j,i])
            self.patches[name] = ps
            self.patchxy[name] = xys
            self.images[name] = image
    
    def fetchExp(self):
        print("Fetching Expression...")
        slides = {name: load_training_data(id=name, device=torch.device('cpu')) for name in self.snames}
        gene_idx_path = "/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/projects/spatial_omics/code/out/filtered_gene_indexes.npy"
        fil_gene_idx = np.load(gene_idx_path)
        gene_ex, gene_ex_binary, targetX, targetY = {}, {}, {}, {}
        for name in self.snames:
            gene_ex_binary, gene_ex = [], []
            ex = numpify(slides[name].gene_ex)
            median = np.median(ex, axis=0)[fil_gene_idx]
            for i in range(len(ex)):
                gene_ex.append(ex[i][fil_gene_idx])
                gene_ex_binary.append(ex[i][fil_gene_idx] > median)
            
            self.gene_ex[name] = gene_ex
            self.med_ex[name] = median
            self.gene_ex_binary[name] = gene_ex_binary
            self.nodexy[name] = [list(a) for a in zip(numpify(slides[name].gene_ex_X), \
                                   numpify(slides[name].gene_ex_Y))]
    
    def load_detections(self):
        print("Loading Detections...")
        storeDir = '/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/projects/spatial_omics/code/GNN/store/'
        with open(storeDir + 'xcoords.pickle', 'rb') as handle:
            detx = pickle.load(handle)
        with open(storeDir + 'ycoords.pickle', 'rb') as handle:
            dety = pickle.load(handle)
        
        if (self.patchxy == {}):
            with open(storeDir + 'patchxy.pickle', 'rb') as handle:
                self.patchxy = pickle.load(handle)
        
        detxy = {}
        for name in self.snames:
            detxy[name] = []
            for patch in range(len(self.patchxy[name])):
                dx, dy = self.patchxy[name][patch]
                for index in range(len(detx[name][patch])):
                    x, y = detx[name][patch][index], dety[name][patch][index]
                    if (x == [] or y == []): continue;
                    detxy[name].append((x+dx, y+dy))
        self.detxy = detxy