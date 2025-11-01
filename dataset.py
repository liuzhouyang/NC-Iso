from torch_geometric.datasets import TUDataset, PPI
import numpy as np
import scipy.stats as scistats
from torch_geometric.utils import from_networkx, to_networkx
from torch.utils.data import Dataset
from torch_geometric.data import Batch
import random
import torch
import networkx as nx
import os

def load_tu_dataset(name, use_node_attr = False):
    """ Load real-world datasets, available in PyTorch Geometric.
         
    """
    print('+-------------------------------------------+\n')
    print('Loading raw dataset : ', name , '\n')
    if name == "enzymes":
        dataset = TUDataset(root="./dataset/ENZYMES",
                            name="ENZYMES", 
                            use_node_attr = use_node_attr)
    elif name == "proteins":
        dataset = TUDataset(root="./dataset/PROTEINS",
                            name="PROTEINS",
                            use_node_attr = use_node_attr)
    elif name == "dd":
        # stuck in data generation
        dataset = TUDataset(root="./dataset/DD",
                            name="DD",
                            use_node_attr = use_node_attr)
    elif name == "cox2":
        dataset = TUDataset(root="./dataset/cox2", 
                            name="COX2",
                            use_node_attr = use_node_attr)
    elif name == "reddit-binary":
        dataset = TUDataset(root="./dataset/REDDIT-BINARY", 
                            name="REDDIT-BINARY",
                            use_node_attr = use_node_attr)
    elif name == "msrc_21":
        dataset = TUDataset(root="./dataset/MSRC_21", 
                            name="MSRC_21",
                            use_node_attr = use_node_attr)
    elif name == "collab":
        dataset = TUDataset(root="./dataset/COLLAB", 
                            name="COLLAB",
                            use_node_attr = use_node_attr)
    elif name == "dblp":
        dataset = TUDataset(root="./dataset/DBLP", 
                            name="DBLP_v1",
                            use_node_attr = use_node_attr)
    elif name == "aids":
        dataset = TUDataset(root="./dataset/AIDS",
                            name="AIDS",
                            use_node_attr = use_node_attr)
    elif name == "ptc_fm":
            dataset = TUDataset(root="./dataset/PTC_FM",
                                name="PTC_FM",
                                use_node_attr = use_node_attr)
    elif name == "ptc_fr":
            dataset = TUDataset(root="./dataset/PTC_FR",
                                name="PTC_FR",
                                use_node_attr = use_node_attr)
    elif name == "ptc_mm":
            dataset = TUDataset(root="./dataset/PTC_MM",
                                name="PTC_MM",
                                use_node_attr = use_node_attr)
    elif name == "ptc_mr":
            dataset = TUDataset(root="./dataset/PTC_MR",
                                name="PTC_MR",
                                use_node_attr = use_node_attr)                        
    elif name == "mutag":
            dataset = TUDataset(root="./dataset/MUTAG",
                                name="MUTAG",
                                use_node_attr = use_node_attr)
    elif name == "firstmm_db":
        dataset = TUDataset(root="./dataset/FIRSTMM_DB",
                            name="FIRSTMM_DB",
                            use_node_attr = use_node_attr)
    elif name == "ppi":
        dataset = PPI(root="./dataset/PPI")
   
    else:
        raise Exception("Error: unrecognized dataset")
    if name == 'ppi':
        pass
    else:
        print('num classes : ', dataset.num_classes)
        print('num node labels : ', dataset.num_node_labels)
        print('num node features : ', dataset.num_node_features)
        print('num node attributes : ', dataset.num_node_attributes)
        print('num edge labels : ', dataset.num_edge_labels)
        print('num edge attributes : ', dataset.num_edge_attributes)
    print('+-------------------------------------------+')
    if dataset[0].x == None:
        num_node_labels = 0
    else:
        num_node_labels = dataset[0].x.size(-1)
    train_len = int(len(dataset) * 0.6)
    val_len = int(len(dataset) * 0.2)
    dataset = dataset.shuffle()
    graphs = []
    for i, data in enumerate(dataset):
        if not type(data) == nx.Graph:
            if num_node_labels == 0:
                graph = to_networkx(data).to_undirected()
            else:
                graph = to_networkx(data, node_attrs = ['x']).to_undirected()
            graphs.append(graph)
    train_len = int(len(dataset) * 0.8 * (1.0 - 0.2))
    val_len = int(len(dataset) * 0.8) - train_len
    return num_node_labels, (graphs[:train_len], graphs[train_len:train_len+val_len], graphs[train_len+val_len:]), \
           (dataset[:train_len], dataset[train_len:train_len+val_len], dataset[train_len+val_len:])

def rooted_random_walk(graphs, sampling_prob, data_size = 0, fix_id = None, task='data', name = None):
    while True:
        idx = fix_id
        if idx is None:
            idx = sampling_prob.rvs()
        else:
            if task == 'neg':
                tmp = idx
                while tmp == idx:
                    tmp = sampling_prob.rvs()
                idx = tmp
        graph = graphs[idx].copy()
        start_node = random.choice(list(graph.nodes))
        num_nodes = graph.number_of_nodes()
        if task == 'data':
            if name in ['firstmm_db', 'collab', 'dd', 'ppi']:
                if num_nodes > 100:
                    num_nodes = 101
                size = random.randint(min(29, int(num_nodes*0.5)), num_nodes-1)
            else:
                size = random.randint(min(29, int(num_nodes*0.5)), num_nodes-1)
        else:
            size = random.randint(5, max(6,int(data_size*0.6)))
        neigh = [start_node]
        frontier = list(set(graph.neighbors(start_node)) - set(neigh))
        visited = set([start_node])
        while len(neigh) < size and frontier:
            new_node = random.choice(list(frontier))
            #new_node = max(sorted(frontier))
            assert new_node not in neigh
            neigh.append(new_node)
            visited.add(new_node)
            frontier += list(graph.neighbors(new_node))
            frontier = [x for x in frontier if x not in visited]
        if len(neigh) == size:
            G = graph.subgraph(neigh)
            return idx, start_node, G

def downsampling(defined_size, data_graph):
    while True:
        graph = data_graph.copy()
        start_node = random.choice(list(graph.nodes))
        num_nodes = graph.number_of_nodes()
        size = defined_size
        neigh = [start_node]
        frontier = list(set(graph.neighbors(start_node)) - set(neigh))
        visited = set([start_node])
        while len(neigh) < size and frontier:
            new_node = random.choice(list(frontier))
            #new_node = max(sorted(frontier))
            assert new_node not in neigh
            neigh.append(new_node)
            visited.add(new_node)
            frontier += list(graph.neighbors(new_node))
            frontier = [x for x in frontier if x not in visited]
        if len(neigh) == size:
            G = graph.subgraph(neigh)
            return G

def discrete_sampling(graphs):
    probs = np.array([len(g) for g in graphs], dtype=float)
    probs /= np.sum(probs)
    dist = scistats.rv_discrete(values = (np.arange(len(graphs)), probs))
    return dist

class SCDataset(Dataset):
    def __init__(self, dataset, graphs, name = None):
        self.name = name
        self.dataset = dataset
        self.graphs = graphs
        self.sampling_prob = discrete_sampling(self.graphs)
     
    def __getitem__(self,_):
        pass
           
    def _get_graph(self):
        while True:
            idx, _, target = rooted_random_walk(self.graphs, self.sampling_prob, name = self.name)
            if nx.is_connected(target) and len(target.edges) > 0 and len(target.nodes) > 5:
                graph = nx.convert_node_labels_to_integers(target)
                return idx, graph

    def _get_triplet(self):
        """Generate one triplet of graphs."""
        idx, g = self._get_graph()
        num_nodes = g.number_of_nodes()
        _, center, pos_g = rooted_random_walk([g], self.sampling_prob, num_nodes, 0, task='pos')
        
        _, _, neg_g = rooted_random_walk(self.graphs, self.sampling_prob, num_nodes, idx, task='neg')
        if g is not None and pos_g is not None and neg_g is not None:
            return center, g, pos_g, neg_g
    
    def get_ranking_graphs(self, num_rank):
        graphs = []
        sizes = []
        while True:
            idx, _, target = rooted_random_walk(self.graphs, self.sampling_prob, name = self.name)
            if nx.is_connected(target) and len(target.edges) > 0 and len(target.nodes) > 6+num_rank:
                data_graph = nx.convert_node_labels_to_integers(target)
                break
        num_nodes = data_graph.number_of_nodes()
        sizes.append(num_nodes)
        nums = np.arange(3,num_nodes)
        num_nodes_subgraph = np.sort(np.random.choice(nums, num_rank, replace=False))[::-1]
        subgraph = data_graph.copy()
        graphs.append(from_networkx(data_graph.copy()))
        for size in num_nodes_subgraph:
            subgraph = nx.convert_node_labels_to_integers(downsampling(size, subgraph))
            graphs.append(from_networkx(subgraph.copy()))
            num_nodes = subgraph.number_of_nodes()
            sizes.append(num_nodes)
        return graphs, sizes
            
    def get_batch(self,batch_size):
        g1s = []
        g2s = []
        g3s = []
        centers_pos= []
        for _ in range(batch_size):
            center, g1, g2, g3 = self._get_triplet()
            centers_pos.append(center)
            g1, g2, g3 = from_networkx(g1), from_networkx(g2), from_networkx(g3)
            if g1.x == None:
                g1.x = torch.tensor(g1.num_nodes * [1.0]).unsqueeze(-1)
                g2.x = torch.tensor(g2.num_nodes * [1.0]).unsqueeze(-1)
                g3.x = torch.tensor(g3.num_nodes * [1.0]).unsqueeze(-1)
            g1s.append(g1)
            g2s.append(g2)
            g3s.append(g3)
        return centers_pos, g1s, g2s, g3s

    def triplets(self, batch_size):
        """Yields batches of triplet data."""
        g1s = []
        g2s = []
        g3s = []
        g1_size = []
        g2_size = []
        g3_size = []
        idx1 = []
        idx2 = []
        idx3 = []
        graphs1 = []
        graphs2 = []
        graphs3 = []
        centers_pos= []
        max_node_g1 = 0
        mean_node_g1 = 0
        min_node_g1 = 10000
        max_node_g2 = 0
        mean_node_g2 = 0
        min_node_g2 = 10000
        max_node_g3 = 0
        mean_node_g3 = 0
        min_node_g3 = 10000

        max_edge_g1 = 0
        mean_edge_g1 = 0
        min_edge_g1 = 10000
        max_edge_g2 = 0
        mean_edge_g2 = 0
        min_edge_g2 = 10000
        max_edge_g3 = 0
        mean_edge_g3 = 0
        min_edge_g3 = 10000
        for _ in range(batch_size):
            center, g1, g2, g3 = self._get_triplet()
            g1_node = g1.number_of_nodes()
            g2_node = g2.number_of_nodes()
            g3_node = g3.number_of_nodes()
            g1_edge = g1.number_of_edges()
            g2_edge = g2.number_of_edges()
            g3_edge = g3.number_of_edges()
            if g1_node > max_node_g1:
                max_node_g1 = g1_node
            if g2_node > max_node_g2:
                max_node_g2 = g2_node
            if g3_node > max_node_g3:
                max_node_g3 = g3_node
            if g1_node < min_node_g1:
                min_node_g1 = g1_node
            if g2_node < min_node_g2:
                min_node_g2 = g2_node
            if g3_node < min_node_g3:
                min_node_g3 = g3_node

            if g1_edge > max_edge_g1:
                max_edge_g1 = g1_edge
            if g2_edge > max_edge_g2:
                max_edge_g2 = g2_edge
            if g3_edge > max_edge_g3:
                max_edge_g3 = g3_edge
            if g1_edge < min_edge_g1:
                min_edge_g1 = g1_edge
            if g2_edge < min_edge_g2:
                min_edge_g2 = g2_edge
            if g3_edge < min_edge_g3:
                min_edge_g3 = g3_edge   
            mean_node_g1 += g1_node
            mean_node_g2 += g2_node
            mean_node_g3 += g3_node
            mean_edge_g1 += g1_edge
            mean_edge_g2 += g2_edge
            mean_edge_g3 += g3_edge
            
            centers_pos.append(center)
            g1, g2, g3 = from_networkx(g1), from_networkx(g2), from_networkx(g3)
            if g1.x == None:
                g1.x = torch.tensor(g1.num_nodes * [1.0]).unsqueeze(-1)
                g2.x = torch.tensor(g2.num_nodes * [1.0]).unsqueeze(-1)
                g3.x = torch.tensor(g3.num_nodes * [1.0]).unsqueeze(-1)
            g1s.append(g1)
            g2s.append(g2)
            g3s.append(g3)
        mean_node_g1 /= batch_size
        mean_node_g2 /= batch_size
        mean_node_g3 /= batch_size
        mean_edge_g1 /= batch_size
        mean_edge_g2 /= batch_size
        mean_edge_g3 /= batch_size
        return centers_pos, g1s, g2s, g3s, \
               max_node_g1, max_node_g2, max_node_g3, \
               mean_node_g1, mean_node_g2, mean_node_g3, \
               min_node_g1, min_node_g2, min_node_g3, \
               max_edge_g1, max_edge_g2, max_edge_g3, \
               mean_edge_g1, mean_edge_g2, mean_edge_g3, \
               min_edge_g1, min_edge_g2, min_edge_g3
    
    def single(self):
        g1, g2, g3 = self._get_triplet()
        g1, g2, g3 = from_networkx(g1), from_networkx(g2), from_networkx(g3)
        if g1.x == None:
                g1.x = torch.tensor(g1.num_nodes * [1.0]).unsqueeze(-1)
                g2.x = torch.tensor(g2.num_nodes * [1.0]).unsqueeze(-1)
                g3.x = torch.tensor(g3.num_nodes * [1.0]).unsqueeze(-1)
        return [g1, g2, g3]

