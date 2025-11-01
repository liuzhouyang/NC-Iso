import torch
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Module, Dropout, Identity, BatchNorm1d, InstanceNorm1d, \
                     LayerNorm, Linear, Sequential, ReLU, ModuleList, RNNCell, \
                     GRUCell
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, \
                               SAGEConv, GINConv
from torch_geometric.utils import remove_self_loops


class norm_act_drop(Module):
    def __init__(self, size: int, norm_module: str, activation: str, \
                       dropout_prob: float, final_layer: bool = False):
        super().__init__()
        self.norm = self.get_norm_layer(size, norm_module) \
                                if norm_module != 'none' else None
        self.activation, self.dropout = None, None
        if not final_layer:
            self.activation = getattr(torch.nn, activation)()
            self.dropout = Dropout(dropout_prob) \
                                        if dropout_prob else None

    @staticmethod
    def get_norm_layer(size, norm_module='none'):
        if norm_module == 'none':
            return Identity()
        elif norm_module == 'batch':
            return BatchNorm1d(size)
        elif norm_module == 'instance':
            return InstanceNorm1d(size)
        elif norm_module == 'layer':
            return LayerNorm(size)
        else:
            return NotImplementedError(f"Not Implemented norm layer {norm_module}")

    def forward(self, x):
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

class MLPGatedConv(MessagePassing):
    def __init__(self, gated, hidden_dim, **kwargs):
        super(MLPGatedConv, self).__init__(aggr='add', **kwargs)
        self.gated = gated
        if self.gated:
            self.rnn = GRUCell(hidden_dim, hidden_dim)
        else:
            self.rnn = RNNCell(hidden_dim, hidden_dim)
        self.nn = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        self.rnn.reset_parameters()
        #self.nn.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        h_ = self.propagate(edge_index, x=x)
        out = self.rnn(x, h_)
        return self.nn(out)

    def message(self, x_j):
        return x_j

    def __repr__(self):
        return '{}(rnn={})'.format(self.__class__.__name__, self.rnn)

class GNN(Module):
    def __init__(self, args):
        super(GNN, self).__init__()
        self.num_layers = args.num_layers

        self.preprocessor = Linear(args.input_dim, args.hidden_dim)
        
        self.convs = ModuleList()
        self.layer_type = args.layer_type
        
        block_layer = self.build_layer(args.layer_type, args.gated, args.hidden_dim)
        for i in range(self.num_layers):
            self.convs.append(block_layer(args.hidden_dim, args.hidden_dim))
        self.skip_connection = args.skip_connection
        self.act = args.encoder_act
        self.norm = args.encoder_norm
        self.dropout = args.encoder_dropout
        self.postprocessor = norm_act_drop(args.hidden_dim, self.norm, self.act, self.dropout)
        if self.skip_connection == 'linear' or self.skip_connection == 'initial':
            self.lin = Linear(args.hidden_dim, args.hidden_dim)
               
    def build_layer(self, layer_type, gated, hidden_dim):
        if layer_type == 'GIN':
            return lambda i, h: GINConv(Sequential(
                Linear(i, h), ReLU(), Linear(h, h)))
        elif layer_type == 'SAGE':
            return SAGEConv
        elif layer_type == 'GCN':
            return GCNConv
        elif layer_type == 'GAT':
            return GATConv
        elif layer_type == 'MLPGated':
            return MLPGatedConv
        else:
            raise NotImplementedError('GNN model not implemented')

    def forward(self, x, edge_index):
        all_emb = None
        # [num_nodes_per_batch, one_hot] --> [num_nodes_per_batch, input_dim]
        x = self.preprocessor(x)
        x0 = x
        for i in range(self.num_layers):
            # [num_nodes_per_batch, input_dim] --> [num_nodes_per_batch, hidden_dim]
            z = self.convs[i](x, edge_index)
            if self.skip_connection == 'linear':
                x = (self.lin(x) + z)
            elif self.skip_connection == 'initial':
                x = (self.lin(x0) + z)
            elif self.skip_connection == 'identity':
                x = x0 + z
                x0 = x
            else:
                x = z
            x = self.postprocessor(x)
            if all_emb == None:
                # [num_nodes_per_batch, hidden_dim] --> [1, num_nodes_per_batch, hidden_dim]
                all_emb = x.unsqueeze(0)
            else:
                # [1, num_nodes_per_batch, hidden_dim] --> [n, num_nodes_per_batch, hidden_dim]
                all_emb = torch.cat((all_emb, x.unsqueeze(0)))
        return all_emb
