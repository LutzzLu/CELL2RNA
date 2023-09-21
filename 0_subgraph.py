import pickle
import os
import numpy as np
import geovoronoi
import shapely
import matplotlib.pyplot as plt
import pandas as pd
import collections
from collections import defaultdict

import torch

from DataManager import dataManager
from graphgen import neighborFinder

from visualize_2 import plotHeatmaps, plotAUROC
import PIL
from tqdm import tqdm, trange

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

def load_cell_data():
    store = '/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/Lutz/CELL2RNA/spatial_omics/code/GNN/store'
   
    with open(f"{store}/xcoords.pickle", "rb") as f:
        xcoords = pickle.load(f)

    with open(f"{store}/ycoords.pickle", "rb") as f:
        ycoords = pickle.load(f)

    with open(f"{store}/patchxy.pickle", 'rb') as f:
        patch_xy = pickle.load(f)
        
    with open(f"{store}/cnn_embed.pickle", "rb") as f:
        embeddings = pickle.load(f)

    slide_cell_locations_all = {}
    slide_embeddings_all = {}
    for slide in ['A1', 'B1', 'C1', 'D1']:
        cell_x = []
        cell_y = []
        embeddings_ = []
        
        xy_set = set()

        cell_counter = 0
        for patch_i, (patch_x, patch_y) in enumerate(patch_xy[slide]):
            for cell_i in range(len(xcoords[slide][patch_i])):
                x = xcoords[slide][patch_i][cell_i] + patch_x
                y = ycoords[slide][patch_i][cell_i] + patch_y
                
                embedding = embeddings[slide][cell_counter]
                
                cell_counter += 1
                
                # Remove duplicates, which does happen sometimes
                if (x, y) in xy_set:
                    continue
                    
                xy_set.add((x, y))

                cell_x.append(x)
                cell_y.append(y)
                embeddings_.append(embedding)
                
        slide_cell_locations_all[slide] = np.vstack([cell_x, cell_y]).T
        slide_embeddings_all[slide] = np.array(embeddings_)
        
    return slide_cell_locations_all, slide_embeddings_all

cell_locations, cell_embeddings = load_cell_data()
with open('/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/Lutz/CELL2RNA/spatial_omics/DH/visium/preprocessed_data/visium_data_filtered_processed.pkl', "rb") as f:
    visium = pd.read_pickle(f)
    
with open('/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/Lutz/CELL2RNA/all_code/adjacency_matrices.pkl', "rb") as f:
    adjacency_matrices = pd.read_pickle(f)
    
    
def create_voronoi_regions(cell_locations):
    min_x = np.min(cell_locations[:, 0])
    max_x = np.max(cell_locations[:, 0])
    min_y = np.min(cell_locations[:, 1])
    max_y = np.max(cell_locations[:, 1])
    
    bounding_rect = shapely.geometry.Polygon([
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
        [max_x, min_y],
    ])
    
    region_polys, region_pts = geovoronoi.voronoi_regions_from_coords(cell_locations, bounding_rect)
    
    filtered_polys = []
    filtered_region_pts = []
    
    for i, polygon in region_polys.items():
        polygon_min_x, polygon_min_y, polygon_max_x, polygon_max_y = polygon.bounds
        if (polygon_min_x == min_x or
            polygon_min_y == min_y or
            polygon_max_x == max_x or
            polygon_max_y == max_y):
            continue
        
        filtered_polys.append(polygon)
        filtered_region_pts.append(region_pts[i])
    
    return filtered_polys, filtered_region_pts

regions = {}
# Create final locations and embeddings dicts, removing droped cells
locations = {}
embeddings = {}

for slide_id in cell_locations.keys():
    regions[slide_id], preserved_indexes = create_voronoi_regions(cell_locations[slide_id])
    locations[slide_id] = cell_locations[slide_id][[x[0] for x in preserved_indexes]]
    embeddings[slide_id] = cell_embeddings[slide_id][[x[0] for x in preserved_indexes]]
    
def get_indexes_of_cells_under_each_spot(slide, cell_locations, r):
    indexes = []
    counts, visium_locations = slide
    r2 = r * r
    
    # Sort cells into tiles
    tiles = defaultdict(set)
    for cell_i, (cell_x, cell_y) in enumerate(cell_locations):
        for dx in [-r, 0, r]:
            for dy in [-r, 0, r]:
                tiles[(cell_x + dx) // (2 * r), (cell_y + dy) // (2 * r)].add(cell_i)

    for visium_i in range(len(visium_locations)):
        indexes.append([])

        visium_x, visium_y = visium_locations.iloc[visium_i]
        
        search_space = set()
        for dx in [-r, 0, r]:
            for dy in [-r, 0, r]:
                search_space = search_space.union(
                    tiles[(visium_x + dx) // (2 * r), (visium_y + dy) // (2 * r)]
                )

        for cell_i in search_space:
            cell_x, cell_y = cell_locations[cell_i]

            distance = ((cell_x - visium_x) ** 2 + (cell_y - visium_y) ** 2)
            if distance <= r2:
                indexes[-1].append(cell_i)
                
    return indexes

cells_by_spot = {}

for slide_id in visium.keys():
    cells_by_spot[slide_id] = get_indexes_of_cells_under_each_spot(visium[slide_id], locations[slide_id], r=55)#55
    
    
    
def create_neighborhood(adjacency_matrix, starting_indexes, hop_count):
    accessible = np.zeros(adjacency_matrix.shape[0], dtype=bool)
    accessible[starting_indexes] = True
    
    for hop in range(hop_count):
        accessible = accessible | (accessible @ adjacency_matrix)
        
    return np.where(accessible)[0]

def create_neighborhood_edge_list(edge_list, starting_indexes, hop_count):
    accessible = set(starting_indexes)
    
    for hop in range(hop_count):
        next_accessible = {*accessible}
        for accessible_index in accessible:
            next_accessible.update(edge_list[accessible_index])
            
        accessible = next_accessible
        
    return np.array([*sorted(accessible)])

# neighborhoods_by_spot = {}
neighborhoods_by_spot_3 = {}
neighborhoods_by_spot_4 = {}
neighborhoods_by_spot_5 = {}
edge_lists = {}

for slide_id in cells_by_spot.keys():
    neighborhoods = []
    adj = adjacency_matrices[slide_id]
    edge_lists[slide_id] = [np.where(adj[i])[0] for i in range(adj.shape[0])]
    for spot_i in tqdm(range(len(cells_by_spot[slide_id])), desc='Creating neighborhoods for ' + str(slide_id)):
        starting_cells = cells_by_spot[slide_id][spot_i]
        
        neighborhoods.append(
            create_neighborhood_edge_list(edge_lists[slide_id], starting_cells, hop_count=3)#3
        )
    
    neighborhoods_by_spot_3[slide_id] = neighborhoods
    
    
for slide_id in cells_by_spot.keys():
    neighborhoods = []
    adj = adjacency_matrices[slide_id]
    edge_lists[slide_id] = [np.where(adj[i])[0] for i in range(adj.shape[0])]
    for spot_i in tqdm(range(len(cells_by_spot[slide_id])), desc='Creating neighborhoods for ' + str(slide_id)):
        starting_cells = cells_by_spot[slide_id][spot_i]
        
        neighborhoods.append(
            create_neighborhood_edge_list(edge_lists[slide_id], starting_cells, hop_count=4)#3
        )
    
    neighborhoods_by_spot_4[slide_id] = neighborhoods
    
    
for slide_id in cells_by_spot.keys():
    neighborhoods = []
    adj = adjacency_matrices[slide_id]
    edge_lists[slide_id] = [np.where(adj[i])[0] for i in range(adj.shape[0])]
    for spot_i in tqdm(range(len(cells_by_spot[slide_id])), desc='Creating neighborhoods for ' + str(slide_id)):
        starting_cells = cells_by_spot[slide_id][spot_i]
        
        neighborhoods.append(
            create_neighborhood_edge_list(edge_lists[slide_id], starting_cells, hop_count=5)#3
        )
    
    neighborhoods_by_spot_5[slide_id] = neighborhoods
    
class CellSubgraphModelV2(pl.LightningModule):
    def __init__(self, n_genes: int, GraphLayer: type, d_model: int, n_layers: int):
        super().__init__()
        
        self.n_genes = n_genes
        self.d_model = d_model
        self.n_layers = n_layers
        
        self.graph_layer_5_4 = GraphLayer(d_model, d_model)
        self.graph_layer_4_3 = GraphLayer(d_model, d_model)
        self.graph_layer_3_1 = GraphLayer(d_model, d_model)
        self.graph_layer_3_2 = GraphLayer(d_model, d_model)
        
#         graph_layers = [
#             GraphLayer(d_model, d_model) for _ in range(n_layers)
#         ]
#         self.graph_layers = nn.ModuleList(graph_layers)
        
        self.gene_head = nn.Linear(d_model, n_genes)
        
        self.save_hyperparameters()
        
    def forward(self, embeddings_3, 
                embeddings_4,
                embeddings_5,
                edge_index_3, 
                edge_index_4, 
                edge_index_5, 
                from_5_4,
                from_4_3,):
        
        embeddings_5 = self.graph_layer_5_4(embeddings_5, edge_index_5)
#         print(embeddings_5)
#         print(from_5_4)
        embeddings_4 = embeddings_5[from_5_4.numpy()[0]]
#         print(embeddings_4)
        embeddings_4 = self.graph_layer_4_3(embeddings_4, edge_index_4)
        embeddings_3 = embeddings_4[from_4_3.numpy()[0]]
        embeddings_3 = self.graph_layer_3_1(embeddings_3, edge_index_3)
        embeddings_3 = self.graph_layer_3_2(embeddings_3, edge_index_3)
        
#         for layer_i in range(self.n_layers):
#             embeddings = self.graph_layers[layer_i](embeddings, edge_index)
        
        # Final activation is softplus, to ensure that the results are positive
        return F.softplus(self.gene_head(embeddings_3))
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
    
    # Predicts log1p. To find raw counts, use expm1 
    def spot_prediction(self, instance):
        embeddings_3, embeddings_4,embeddings_5,edge_index_3, edge_index_4, edge_index_5, from_5_4,from_4_3, covered_indexes, counts = instance
        
        cell_predictions = self.forward(embeddings_3, 
                                        embeddings_4,
                                        embeddings_5,
                                        edge_index_3, 
                                        edge_index_4, 
                                        edge_index_5, 
                                        from_5_4,
                                        from_4_3,)
        spot_prediction = cell_predictions[covered_indexes].mean(dim=0)
        
        return spot_prediction
    
    def training_step(self, batch, batch_idx):
        embeddings_3, embeddings_4,embeddings_5,edge_index_3, edge_index_4, edge_index_5, from_5_4,from_4_3, covered_indexes, counts = instance = [x[0] for x in batch]
        
        loss = F.mse_loss(torch.log(1 + self.spot_prediction(instance)), torch.log(1 + counts))
        
        self.log('train_loss', loss.item(), prog_bar=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        embeddings_3, embeddings_4,embeddings_5,edge_index_3, edge_index_4, edge_index_5, from_5_4,from_4_3, covered_indexes, counts = instance = [x[0] for x in batch]
        
        loss = F.mse_loss(torch.log(1 + self.spot_prediction(instance)), torch.log(1 + counts))
        
        self.log('validation_loss', loss.item(), prog_bar=True, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        embeddings_3, embeddings_4,embeddings_5,edge_index_3, edge_index_4, edge_index_5, from_5_4,from_4_3, covered_indexes, counts = instance = [x[0] for x in batch]
        
        loss = F.mse_loss(torch.log(1 + self.spot_prediction(instance)), torch.log(1 + counts))
        
        self.log('test_loss', loss.item(), prog_bar=True, on_epoch=True)
        
        return loss
    
    def predict_step(self, batch, batch_idx):
        return self.spot_prediction([x[0] for x in batch])
    
import torch.utils.data

def ensure_tensorlist(L):
    return [ensure_tensor(x) for x in L]

def ensure_tensor(x):
    return torch.tensor(x) if type(x) is not torch.Tensor else x

class CellSubgraphDataset(torch.utils.data.Dataset):
    """
    One slide at a time.
    
    Parameters:
    - cell_embeddings: torch.Tensor of cell embeddings
    - cells_by_spot: List of cell indexes corresponding to Visium node
    - neighborhoods_by_spot: List of cell indexes corresponding to Visium node
    - adjacency_matrix: torch.Tensor containing adjacency matrix (cell -> cell)
    - counts: torch.Tensor containing gene counts of each Visium node
    """
    
    def __init__(self, cell_embeddings, cells_by_spot, 
                 neighborhoods_by_spot_3, 
                 neighborhoods_by_spot_4, 
                 neighborhoods_by_spot_5, 
                 adjacency_matrix, counts):
        self.cells_by_spot = ensure_tensorlist(cells_by_spot)
        self.neighborhoods_by_spot_3 = ensure_tensorlist(neighborhoods_by_spot_3)
        self.neighborhoods_by_spot_4 = ensure_tensorlist(neighborhoods_by_spot_4)
        self.neighborhoods_by_spot_5 = ensure_tensorlist(neighborhoods_by_spot_5)
        self.cell_embeddings = ensure_tensor(cell_embeddings)
        self.adjacency_matrix = ensure_tensor(adjacency_matrix)
        self.counts = ensure_tensor(counts).type(torch.float)

        assert len(self.cells_by_spot) == len(self.neighborhoods_by_spot_3)
        assert len(cell_embeddings) == adjacency_matrix.shape[0]
        
    def __len__(self):
        return len(self.cells_by_spot)
    
    def __getitem__(self, index):
        cells_in_spot = self.cells_by_spot[index]
        cells_in_neighborhood_3 = self.neighborhoods_by_spot_3[index]
        cells_in_neighborhood_4 = self.neighborhoods_by_spot_4[index]
        cells_in_neighborhood_5 = self.neighborhoods_by_spot_5[index]
        
        # Re-index cells. Treat cells_in_neighborhood as the universal set. Create an edge list.
        # https://stackoverflow.com/questions/22927181/selecting-specific-rows-and-columns-from-numpy-array
        adjacency_matrix_subgraph_3 = self.adjacency_matrix[cells_in_neighborhood_3, :][:, cells_in_neighborhood_3]
        adjacency_matrix_subgraph_4 = self.adjacency_matrix[cells_in_neighborhood_4, :][:, cells_in_neighborhood_4]
        adjacency_matrix_subgraph_5 = self.adjacency_matrix[cells_in_neighborhood_5, :][:, cells_in_neighborhood_5]
        
        # tr = transformed
        cells_in_neighborhood_tr = torch.arange(len(cells_in_neighborhood_3))
        
        cell_embeddings_3 = self.cell_embeddings[cells_in_neighborhood_3]
        cell_embeddings_4 = self.cell_embeddings[cells_in_neighborhood_4]
        cell_embeddings_5 = self.cell_embeddings[cells_in_neighborhood_5]
        
#         print(cells_in_neighborhood_5)
        
        from_5_4 = []
        from_4_3 = []
        
        for one_index in range(cells_in_neighborhood_5.numpy().shape[0]):
            one = cells_in_neighborhood_5.numpy()[one_index]
            if one in cells_in_neighborhood_4.numpy():
                from_5_4.append(one_index)
                
        for one_index in range(cells_in_neighborhood_4.numpy().shape[0]):
            one = cells_in_neighborhood_4.numpy()[one_index]
            if one in cells_in_neighborhood_3.numpy():
                from_4_3.append(one_index)
                
        from_5_4 = torch.tensor(np.array(from_5_4)).reshape(1, len(from_5_4))
        from_4_3 = torch.tensor(np.array(from_4_3)).reshape(1, len(from_4_3))
        
        cells_in_spot_tr = np.array([
            # Find index within cells_in_neighborhood of corresponding index
            # np.where returns a tuple of parallel arrays. In this case, it is
            # a tuple of length 1 with an array of length 1.
            torch.where(cells_in_neighborhood_3 == cell_in_spot)[0][0]
            for cell_in_spot in cells_in_spot
        ])
        
        edge_index_3 = torch.stack(torch.where(adjacency_matrix_subgraph_3))
        edge_index_4 = torch.stack(torch.where(adjacency_matrix_subgraph_4))
        edge_index_5 = torch.stack(torch.where(adjacency_matrix_subgraph_4))
        
        counts = self.counts[index]
        
        return (cell_embeddings_3, 
                cell_embeddings_4, 
                cell_embeddings_5, 
                edge_index_3, 
                edge_index_4, 
                edge_index_5, 
#                 cells_in_neighborhood_5,
                from_5_4,
                from_4_3,
                cells_in_spot_tr, counts)

    
datasets = {}

for slide_id in visium.keys():
# for slide_id in ['A1']:
    counts, locations = visium[slide_id]
    counts_tensor = torch.tensor(counts.values)
    
    included_spots = {i for i in range(counts.shape[0]) if len(cells_by_spot[slide_id][i]) > 0}
    
    datasets[slide_id] = CellSubgraphDataset(
        embeddings[slide_id],
        [x for i, x in enumerate(cells_by_spot[slide_id]) if i in included_spots],
        [x for i, x in enumerate(neighborhoods_by_spot_3[slide_id]) if i in included_spots],
        [x for i, x in enumerate(neighborhoods_by_spot_4[slide_id]) if i in included_spots],
        [x for i, x in enumerate(neighborhoods_by_spot_5[slide_id]) if i in included_spots],
        adjacency_matrices[slide_id],
        torch.stack([
            counts_tensor[i] for i in range(counts_tensor.shape[0]) if i in included_spots
        ]),
    )
    
import torch_geometric as pyg

train_dataset = torch.utils.data.ConcatDataset([
    datasets['A1'],
    datasets['B1'],
    datasets['C1'],
])

valid_dataset = datasets['D1']

train_loader = torch.utils.data.DataLoader(train_dataset, 1, shuffle=True, num_workers=1)
valid_loader = torch.utils.data.DataLoader(valid_dataset, 1, num_workers=1)

# 3 layers for 3 hops
model = CellSubgraphModelV2(n_genes=17943, GraphLayer=pyg.nn.GATConv, d_model=1000, n_layers=3)

trainer = pl.Trainer(
    max_epochs=32,
)

trainer.fit(model, train_loader, valid_loader)