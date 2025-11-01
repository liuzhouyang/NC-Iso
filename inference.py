import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, recall_score
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score, precision_score
from torch_geometric.nn.glob import global_add_pool, global_mean_pool, global_max_pool
import os
    
def evaluator(pred, raw_pred, labels):
    auroc = roc_auc_score(labels, raw_pred)
    if auroc < 0.5:
        raw_pred = -raw_pred
        auroc = roc_auc_score(labels, raw_pred)
    acc = accuracy_score(labels, pred)
    prec = precision_score(labels, pred)
    recall = recall_score(labels, pred)
    avg_prec = average_precision_score(labels, raw_pred)
    tn, fp, fn, tp = confusion_matrix(labels, pred, normalize = 'all').ravel()
    results = {f'auroc': auroc,
               f'acc':acc,
               f'prec':prec,
               f'recall':recall,
               f'avg_prec':avg_prec,
               f'tn':tn,
               f'fp':fp,
               f'fn':fn,
               f'tp':tp}
    return results
    
def preprocess(g1s, g2s, g3s):
    g1s_nodes = 0
    g2s_nodes = 0
    g3s_nodes = 0
    for g1, g2, g3 in zip(g1s, g2s, g3s):
        g1s_nodes += g1.x.size(0)
        g2s_nodes += g2.x.size(0)
        g3s_nodes += g3.x.size(0)
    return g1s_nodes, g2s_nodes, g3s_nodes,

def get_data(ind, path):
    [g1s, g2s, g3s] = torch.load(f'{path}batch_{ind[0].item()}.pt')
    return g1s, g2s, g3s

@torch.no_grad()
def get_preds(model, test, seed, batches, device, args):
    preds = []
    raw_preds = []
    runtime = 0.0
    #batches = []
    if test:
        num_batches = 2048
        path = 'testset'
    else:
        num_batches = 1024
        path = 'valset'
    all_labels = torch.tensor(([1]*args.batch_size + [0] * args.batch_size) *\
                                    num_batches).to(device)
    for i in range(num_batches):
        [_, g1, g2, g3] = torch.load(f'../datasetStorage/NC/{path}/{args.dataset_name}/{seed}/batch_{i}.pt')
        
        g1s_nodes, g2s_nodes, g3s_nodes = preprocess(g1, g2, g3)
        D_batch = Batch.from_data_list(g1+g1).batch.to(device)
        Q_batch = Batch.from_data_list(g2+g3).batch.to(device)
        data = Batch.from_data_list(g1+g2+g3).to(device)
        t0 = time.time()
        
        raw_node_embs, raw_graph_embs = model(data.x, data.edge_index, data.batch)
        
        g2g_score = torch.ones((args.batch_size*2, 1)).cuda()
        n2g_score = torch.ones((args.batch_size*2, 1)).cuda()
        # in case of single scale
        if len(raw_graph_embs.size()) < 3:
            raw_graph_embs = raw_graph_embs.unsqueeze(0)
        raw_graph_embs = raw_graph_embs.permute(1,0,-1)
        Q_embs = raw_graph_embs[args.batch_size:]
        
        if len(raw_node_embs.size()) < 3:
            raw_node_embs = raw_node_embs.unsqueeze(0)
        # [num_layers, num_nodes, hidden_dim] -> [num_nodes, num_layers, hidden_dim]
        raw_node_embs = raw_node_embs.permute(1,0,-1)
    
        # [num_data_nodes, num_layers, output_dim], [num_query_nodes, num_layers, output_dim]
        g1_nodes = raw_node_embs[:g1s_nodes]
        # prepare for pooling
        nodes_D = torch.cat((g1_nodes, g1_nodes), dim = 0)
        [num_nodes, num_layers, dim] = nodes_D.shape
        nodes_D = torch.reshape(nodes_D, (num_nodes, num_layers * dim))
        
        g = global_max_pool(nodes_D, D_batch)
        g = torch.reshape(g, (num_graphs, self.num_layers, -1))
        [num_graphs, _] = g.shape
        D_embs = torch.reshape(g, (num_graphs, self.num_layers, -1))
        
        score = model.supervision(D_embs, Q_embs)
        raw_pred = score.squeeze()
        
        pred = model.predictor(raw_pred.unsqueeze(-1)).argmax(dim = -1)
        runtime += time.time() - t0
        preds.append(pred.squeeze())
        raw_preds.append(raw_pred.squeeze())
        all_labels.append(labels)

    pred = torch.cat(preds, dim = -1)
    raw_pred = torch.cat(raw_preds, dim = -1)
    labels = torch.cat(all_labels, dim = -1)
    labels = labels.detach().cpu().numpy()
    raw_pred = raw_pred.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    
    results = evaluator(pred, raw_pred, labels)
    return results, runtime

@torch.no_grad()
def test(model, testset, device, args):
    model.eval()
    result, runtime = get_preds(model, True, testset, device, args)
    return result, runtime

@torch.no_grad()
def validation(model, valset, device, args):
    model.eval()
    result, runtime = get_preds(model, False, valset, device, args)
    return result, runtime
