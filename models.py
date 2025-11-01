#---------------------------pytorch----------------------------------
import torch
from torch.nn import Module, Parameter, Linear, Sequential, Dropout, GRU
from torch_geometric.utils import to_dense_batch, scatter, add_self_loops
import torch.nn.functional as F
import random
from torch_geometric.nn.glob import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_batch
from layers import GNN, norm_act_drop

class NC(Module):
    def __init__(self, args):
        super(NC, self).__init__()
        self.gnn = GNN(args)
        self.fc = Sequential(Linear(args.hidden_dim, int(args.hidden_dim / 2)),
                              norm_act_drop(int(args.hidden_dim / 2), args.norm, args.act, args.dropout),
                              Linear(int(args.hidden_dim / 2), args.output_dim))
        self.projector = Linear(args.output_dim, args.output_dim)
        self.predictor = Linear(1, 2)
        self.learnable_weight = args.learnable_weight
        self.margin = args.margin
        self.sed_sim = args.sed_sim
        self.iou_sim = args.iou_sim
        self.num_layers = args.num_layers
        self.multi_scale = args.multi_scale
        
        if args.weighted_loss:
            self.gamma = Parameter(torch.ones(1))
        else:
            self.gamma = torch.ones(1).cuda()
        if args.init_hidden:
            if args.input_dim == 1:
                input_dim = 10
            else:
                input_dim = args.input_dim
            self.hidden_init = Linear(input_dim, args.output_dim)
        
        if self.learnable_weight:
            self.weights = Parameter(torch.ones([args.num_layers]))
        else:
            self.weights = torch.ones([args.num_layers]).cuda()

    def forward(self, x, edge_index, batch):
        # generate multi-scale node embedding xs
        # [num_nodes, input_dim] -> [num_layers, num_nodes, hidden_dim]
        xs = self.gnn(x, edge_index)
        
        # reduce dimensionality
        # [num_layers, num_nodes, hidden_dim] -> [num_layers, num_nodes, output_dim]
        hs = self.fc(xs)
        
        # [num_layers, num_nodes, output_dim] -> [num_layers, num_graphs, output_dim]
        g = global_max_pool(hs, batch)
        g = self.projector(g)
        
        # return reduced multi-scale node representations hs and graph embeddings g
        return self.projector(hs), g

    def cal_violation(self, g1, g2):
        violation = torch.max(torch.zeros_like(g1), 
                          g2 - g1)**2
        return violation
    
    def cal_iou_sim(self, g1, g2):
        g1 = torch.max(torch.zeros_like(g1)+1e-7, g1)
        g2 = torch.max(torch.zeros_like(g2)+1e-7, g2)
        intersection = torch.min(g1,g2)
        convex = torch.max(g1,g2)
        sim = intersection / g1 - self.gamma * (convex - g1) / convex
        return sim

    def supervision(self, D_embs, Q_embs):
        [num_graphs, num_layers, dim]= D_embs.shape
        # [num_graphs, num_layers, dim]= Q_embs.shape
        if self.multi_scale:
            D_embs = torch.reshape(D_embs, (num_graphs, num_layers * dim))
            if Q_embs.dim() > 2:
                Q_embs = torch.reshape(Q_embs, (num_graphs, num_layers * dim))
        else:
            D_embs = torch.mean(D_embs, dim = 1)
            if Q_embs.dim() < 3:
                Q_embs = torch.reshape(Q_embs, (num_graphs, num_layers, dim))
            Q_embs = torch.mean(Q_embs, dim = 1)
        # Q_embs = torch.reshape(Q_embs, (num_graphs, num_layers * dim))
        
        if self.sed_sim:
            sed = self.cal_violation(D_embs, Q_embs)
            # or mean?
            if self.multi_scale:
                sed = torch.reshape(sed, (num_graphs, num_layers, dim)).sum(-1)
            else:
                sed = sed.sum(-1)
            sed_sim = torch.exp(-sed)
        else:
            if self.multi_scale:
                sed_sim = torch.ones((num_graphs, num_layers, 1)).cuda()
            else:
                sed_sim = torch.ones((num_graphs, 1)).cuda()
            
        #pos_sed_sim, neg_sed_sim = torch.split(sed_sim, int(num_graphs/2), dim=0)

        if self.iou_sim:
            iou_sim = self.cal_iou_sim(D_embs, Q_embs)
            if self.multi_scale:
                iou_sim = torch.reshape(iou_sim, (num_graphs, num_layers, dim)).mean(-1)
            else:
                iou_sim = iou_sim.mean(-1)
        else:
            if self.multi_scale:
                iou_sim = torch.ones((num_graphs, num_layers, 1)).cuda()
            else:
                iou_sim = torch.ones((num_graphs, 1)).cuda()
        if self.multi_scale:
            return (self.weights * sed_sim.squeeze() * iou_sim.squeeze()).sum(-1)
        return sed_sim.squeeze() * iou_sim.squeeze()
