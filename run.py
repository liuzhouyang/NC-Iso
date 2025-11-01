import torch
import random
import numpy as np

#---------------------------pytorch----------------------------------
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
#----------------------------others----------------------------------
import os
import time
from datetime import datetime
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch, degree
from torch_geometric.nn.glob import global_add_pool, global_mean_pool, global_max_pool
import yaml
import argparse
import time
import wandb
#--------------------------lib---------------------------------------
from dataset import load_tu_dataset, SCDataset

from models import NC
from inference import validation, test


def str2bool(v):
    return v.lower() in ("true", "1")

torch.set_printoptions(precision=4)
gpu = 'cuda:0'
device = torch.device(gpu) if torch.cuda.is_available() \
                              else torch.device("cpu")
                              
def set_seeds_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def log_builder(args):
    
    record_keys = ['name', 'dataset_name', 'task','layer_type']
    comment = ".".join(["{}={}".format(k, v) \
              for k, v in vars(args).items() if k in record_keys])
    current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
    logd = f'./exp_{args.task}/{args.name}_{args.layer_type}_{args.dataset_name}_{current_time}'
    logger = SummaryWriter(log_dir = logd, comment = comment)
    config = vars(args)
    config["path"] = logd
    config_file_name = "config.yaml"
    with open(os.path.join(logd, config_file_name), "w") as file:
        file.write(yaml.dump(config))

    return logger, logd

def log_train(logger, loss, acc, num_batches):
    logger.add_scalar(f"loss/train", loss.item(), num_batches)
    logger.add_scalar(f"acc/train", acc.item(), num_batches)

def preprocess(g1s, g2s, g3s):
    g1s_nodes = 0
    g2s_nodes = 0
    g3s_nodes = 0
    for g1, g2, g3 in zip(g1s, g2s, g3s):
        g1s_nodes += g1.x.size(0)
        g2s_nodes += g2.x.size(0)
        g3s_nodes += g3.x.size(0)
    return g1s_nodes, g2s_nodes, g3s_nodes, 
     
def train(model, optimizer, pred_optimizer, dataset, logger, epoch, args):
    model.train()
    total_loss = 0.0
    labels = torch.tensor(([1]*args.batch_size+[0]*args.batch_size)).to(device)
    runtime = 0.0
    for i in range(args.iterations):
        model.zero_grad()
        pos_centers, g1, g2, g3= dataset.get_batch(args.batch_size)
        g1s_nodes, g2s_nodes, g3s_nodes = preprocess(g1, g2, g3)
        D_batch = Batch.from_data_list(g1+g1).batch.to(device)
        start_time = time.time()
        data = Batch.from_data_list(g1+g2+g3).to(device)
        # returns: [num_layers, num_nodes, hidden_dim],
        #          [num_layers, num_graphs, output_dim]
        raw_node_embs, raw_graph_embs = model(data.x, data.edge_index, data.batch)
        
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
        loss = F.mse_loss(score.squeeze(), labels.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            raw_pred = score.squeeze()
        model.predictor.zero_grad()
        criterion = CrossEntropyLoss()
        pred = model.predictor(raw_pred.unsqueeze(-1))
        pred_loss = criterion(pred, labels)
        pred_loss.backward()
        pred_optimizer.step()
        acc = torch.mean((pred.argmax(dim = -1) == labels).type(torch.float))
        log_train(logger, loss, acc, i+args.iterations*(epoch))
        del data
        runtime += time.time() - start_time
        print('Iteration: {:02d}. time: {:.4f}. loss: {:.4f}. Acc: {:.4f}.'.format(
            i+args.iterations*(epoch), time.time() - start_time, loss.item(), acc.item()),
            end = '                    \r'
            )
        if args.wandb:
            res_dic = {f'loss/train': loss.item(),
                       f'acc/train': graph_acc.item(),
                       f'runtime/train': time.time() - start_time}
            wandb.log(res_dic)
        total_loss += loss
    return total_loss / args.iterations, runtime


def run(args):
    print(f"executing on {device}")
    results_list = []
    for run in range(args.runs):
        if args.wandb:
            wandb.init(project='NC', 
                   name = f'./exp_{args.task}_{args.name}_{args.dataset_name}_{args.seed}', 
                   sync_tensorboard=False)
        set_seeds_all(args.seed[run])
        print(args)

        num_node_labels, \
                (graphs_train, _, _), \
                    (trainset, _, _) = load_tu_dataset(args.dataset_name)
        else:
            pass
        if num_node_labels == 0:
            args.input_dim = 1
        else:
            args.input_dim = num_node_labels
        trainset = SCDataset(trainset, graphs_train, args.dataset_name)
        logger, logd = log_builder(args)
        model = NC(args)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
        pred_optimizer = torch.optim.Adam(model.predictor.parameters(), lr = args.lr)
        print(f'running repetition {run}')
        t0 = time.time()
        bad_epoch = 0
        best_epoch = -1
        best_auroc = float('inf')
        best_acc = float('inf')
        best_prec = float('inf')
        best_recall = float('inf')
        best_avg_prec = float('inf')
        best_tn = float('inf')
        best_fp = float('inf')
        best_fn = float('inf')
        best_tp = float('inf')

        total_runtime = 0.0
        for epoch in range(args.epochs):
            loss, runtime = train(model, optimizer, pred_optimizer, trainset, logger, epoch, args)
            total_runtime += runtime
            logger.add_scalar(f'total_loss/train', loss.item(), epoch)
            logger.add_scalar(f'runtime/train', runtime, epoch)
            if (epoch + 1) % args.eval_steps == 0 and (epoch + 1) >= args.eval_interval:
                result, runtime = validation(model, args.seed[run], device, args)
                total_runtime += runtime
                if best_auroc < result['auroc']:
                    best_auroc = result['auroc']
                    best_acc = result['acc']
                    best_prec = result['prec']
                    best_recall = result['recall']
                    best_avg_prec = result['avg_prec']
                    best_tn = result['tn']
                    best_fp = result['fp']
                    best_fn = result['fn']
                    best_tp = result['tp']
                    
                    best_epoch = epoch
                    bad_epoch = 0
                    if args.save_model:
                        path = f'{logd}/{args.name}_{args.dataset_name}_epoch{epoch}_{args.task}_auroc={best_auroc:.4f}_acc={best_acc:.4f}.pt'
                        torch.save(model.state_dict(), path)
                        path = f'{logd}/{args.name}_{args.dataset_name}_best.pt'
                        torch.save(model.state_dict(), path)
                else:
                    bad_epoch += 1
                    
                logger.add_scalar(f"{args.task}_auroc/val", result['auroc'], epoch)
                logger.add_scalar(f"{args.task}_acc/val", result['acc'], epoch)
                logger.add_scalar(f"{args.task}_prec/val", result['prec'], epoch)
                logger.add_scalar(f"{args.task}_recall/val", result['recall'], epoch)
                logger.add_scalar(f"{args.task}_avg_prec/val", result['avg_prec'], epoch)
                logger.add_scalar(f"{args.task}_tn/val", result['tn'], epoch)
                logger.add_scalar(f"{args.task}_fp/val", result['fp'], epoch)
                logger.add_scalar(f"{args.task}_fn/val", result['fn'], epoch)
                logger.add_scalar(f"{args.task}_tp/val", result['tp'], epoch)
                logger.add_scalar(f"{args.task}_best_auroc/val", best_auroc, epoch)
                logger.add_scalar(f"{args.task}_best_acc/val", best_acc, epoch)
                logger.add_scalar(f"{args.task}_best_prec/val", best_prec, epoch)
                logger.add_scalar(f"{args.task}_best_recall/val", best_recall, epoch)
                logger.add_scalar(f"{args.task}_best_avg_prec/val", best_avg_prec, epoch)
                logger.add_scalar(f"{args.task}_best_tn/val", best_tn, epoch)
                logger.add_scalar(f"{args.task}_best_fp/val", best_fp, epoch)
                logger.add_scalar(f"{args.task}_best_fn/val", best_fn, epoch)
                logger.add_scalar(f"{args.task}_best_tp/val", best_tp, epoch)
                
                logger.add_scalar(f"{args.task}_best_epoch/val", best_epoch, epoch)
                logger.add_scalar(f"{args.task}_runtime/val", runtime, epoch)
                if args.wandb:
                    res_dic = {f'{args.task}_total_loss/train': loss.item(),
                               f'{args.task}_auroc/val': result['auroc'],
                               f'{args.task}_acc/val': result['acc'],
                               f'{args.task}_prec/val': result['prec'],
                               f'{args.task}_recall/val': result['recall'],
                               f'{args.task}_avg_prec/val': result['avg_prec'],
                               f'{args.task}_tn/val': result['tn'],
                               f'{args.task}_fp/val': result['fp'],
                               f'{args.task}_fn/val': result['fn'],
                               f'{args.task}_tp/val': result['tp'],
                               f'{args.task}_best_auroc/val': best_auroc,
                               f'{args.task}_best_acc/val': best_acc,
                               f'{args.task}_best_prec/val': best_prec,
                               f'{args.task}_best_recall/val': best_recall,
                               f'{args.task}_best_avg_prec/val': best_avg_prec,
                               f'{args.task}_best_tn/val': best_tn,
                               f'{args.task}_best_fp/val': best_fp,
                               f'{args.task}_best_fn/val': best_fn,
                               f'{args.task}_best_tp/val': best_tp,
                               f'{args.task}_best_epoch/val': best_epoch,
                               f'{args.task}_runtime/val': runtime}
                    wandb.log(res_dic)
                print('\nEpoch: {:02d}. Loss: {:.4f}.\n'
                      'AUROC: {:.4f}. Best AUROC: {:.4f}. \n'
                      'ACC: {:.4f}. Best ACC: {:.4f}. Best epoch: {:02d}\n'.format(
                        epoch, loss.item(), 
                        result['auroc'], best_auroc, 
                        result['acc'], best_acc, best_epoch), end='               \r')
            if bad_epoch >= args.patience:
                break
            elif total_runtime // 3600 >= 8:
                print('\nTimeout!\n')
                break
        path = f'{logd}/{args.name}_{args.dataset_name}_best.pt'
        del model
        model = NC(args).cuda()
        model.load_state_dict(torch.load(path))
        result, runtime = test(model, args.seed[run], device, args)
        logger.add_scalar(f"{args.task}_runtime/test", runtime)
        logger.add_scalar(f"{args.task}_auroc/test", result['auroc'])
        logger.add_scalar(f"{args.task}_acc/test", result['acc'])
        logger.add_scalar(f"{args.task}_prec/test", result['prec'])
        logger.add_scalar(f"{args.task}_recall/test", result['recall'])
        logger.add_scalar(f"{args.task}_avg_prec/test", result['avg_prec'])
        logger.add_scalar(f"{args.task}_tn/test", result['tn'])
        logger.add_scalar(f"{args.task}_fp/test", result['fp'])
        logger.add_scalar(f"{args.task}_fn/test", result['fn'])
        logger.add_scalar(f"{args.task}_tp/test", result['tp'])

        res_dic = {f'{args.task}_auroc/test': result['auroc'],
                               f'{args.task}_acc/test': result['acc'],
                               f'{args.task}_prec/test': result['prec'],
                               f'{args.task}_recall/test': result['recall'],
                               f'{args.task}_avg_prec/test': result['avg_prec'],
                               f'{args.task}_tn/test': result['tn'],
                               f'{args.task}_fp/test': result['fp'],
                               f'{args.task}_fn/test': result['fn'],
                               f'{args.task}_tp/test': result['tp'],
                               f'{args.task}_runtime/test': runtime}
        if args.wandb:
            wandb.log(res_dic)
        print('\nRun: {:02d}. Runtime: {:.4f}. \n'
              'Best AUROC: {:.4f}. Best ACC: {:.4f}. \n'
              'Test AUROC: {:.4f}. Test ACC: {:.4f}. \n'
              'Best PREC: {:.4f}. Best RECALL: {:.4f}. Best AVG PREC: {:.4f}. \n'
              'Test PREC: {:.4f}. Test RECALL: {:.4f}. Test AVG PREC: {:.4f}. \n'
              'Best TN: {:.4f}. Best FP: {:.4f}. Best FN: {:.4f}. Best TP: {:.4f}. \n'
              'Test TN: {:.4f}. Test FP: {:.4f}. Test FN: {:.4f}. Test TP: {:.4f}. \n'
              'Inference time: {:.4f}.\n'.format(
                run, time.time() - t0, 
                best_auroc, best_acc, 
                result['auroc'], result['acc'], 
                best_prec, best_recall, best_avg_prec, 
                result['prec'], result['recall'], result['avg_prec'],
                best_tn, best_fp, best_fn, best_tp,
                result['tn'], result['fp'], result['fn'], result['tp'],
                runtime
            ),end = '                    \r')
        logger.add_scalar(f"memory_allocated", torch.cuda.memory_allocated())
        logger.add_scalar(f"memory_cached", torch.cuda.memory_reserved())
        torch.cuda.empty_cache()
        filename = f'{logd}/result'
        with open(filename, 'a') as writefile:
            for key, value in result.items():
                writefile.write(key + ' ' + str(value) +'\n')

        tresult = [result['auroc'], result['acc'], result['prec'], result['recall'], result['avg_prec'],\
                   result['tn'], result['fp'], result['fn'], result['tp']]
        path = f'{logd}/result.pt'
        torch.save(tresult, path)
        results_list.append(tresult)
        if args.runs == run + 1:
            mean_auroc, mean_acc, mean_prec, mean_recall, mean_avg_prec, \
                mean_tn, mean_fp, mean_fn, mean_tp = np.mean(results_list, axis=0)
            var = np.var(results_list, axis=0)
            auroc_std = np.sqrt(var[0])
            acc_std = np.sqrt(var[1])
            prec_std = np.sqrt(var[2])
            recall_std = np.sqrt(var[3])
            avg_prec_std = np.sqrt(var[4])
            tn_std = np.sqrt(var[5])
            fp_std = np.sqrt(var[6])
            fn_std = np.sqrt(var[7])
            tp_std = np.sqrt(var[8])
            
            final_result = {f'{args.task}_auroc_mean': mean_auroc, 
                            f'{args.task}_auroc_std': auroc_std,
                            f'{args.task}_acc_mean': mean_acc, 
                            f'{args.task}_acc_std': acc_std,
                            f'{args.task}_prec_mean': mean_prec, 
                            f'{args.task}_prec_std': prec_std,
                            f'{args.task}_recall_mean': mean_recall, 
                            f'{args.task}_recall_std': recall_std,
                            f'{args.task}_avg_prec_mean': mean_avg_prec, 
                            f'{args.task}_avg_prec_std': avg_prec_std,
                            f'{args.task}_tn_mean': mean_tn, 
                            f'{args.task}_tn_std': tn_std,
                            f'{args.task}_fp_mean': mean_fp, 
                            f'{args.task}_fp_std': fp_std,
                            f'{args.task}_fn_mean': mean_fn, 
                            f'{args.task}_fn_std': fn_std,
                            f'{args.task}_tp_mean': mean_tp, 
                            f'{args.task}_tp_std': tp_std}
            print(final_result)
            if args.wandb:
                wandb.log(final_result)
            filename = f'{logd}/results'
            with open(filename, 'a') as writefile:

                for key, value in final_result.items():
                    writefile.write(key + ' ' + str(value) +'\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NC')
   
    parser.add_argument('--dataset_name', type = str, default = 'cox2',
                    help = 'name of the dataset.')
    parser.add_argument('--seed', type = int, default = 4538,
                        help = 'Random seed number.')
    parser.add_argument('--iterations', type = int, default = 100)                  
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--epochs', type = int, default = 5000)
    parser.add_argument('--patience', type = int, default = 50)
    parser.add_argument('--save_train', type = str2bool, default = True)
    
    parser.add_argument('--runs', type=int, default=1, help='the number of repetition of the experiment to run')
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--save_model', type=str2bool, default=True)
    parser.add_argument('--wandb', type = str2bool, default = False)
    parser.add_argument('--task', type = str, default = 'main')
    
    parser.add_argument('--name', type = str, default = 'ncV5')
    parser.add_argument('--weight_decay', type = float, default = 0.0)
    parser.add_argument('--lr', type = float, default = 1e-3,
                    help = 'Learning rate.')
    parser.add_argument('--num_layers', type=int, default= 6)
    parser.add_argument('--hidden_dim', type=int, default= 64)
    parser.add_argument('--output_dim', type=int, default=32)
    parser.add_argument('--learnable_weight', type=str2bool, default=False)
    parser.add_argument('--layer_type', type=str, default='MLPGated')
    parser.add_argument('--skip_connection', type=str, default='none')
    parser.add_argument('--gated', type=str2bool, default=True)
    parser.add_argument('--encoder_act', type=str, default='ReLU')
    parser.add_argument('--encoder_norm', type=str, default='layer')
    parser.add_argument('--encoder_dropout', type=float, default=0.0)
    parser.add_argument('--act', type=str, default='ReLU')
    parser.add_argument('--norm', type=str, default='layer')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--margin', type=float, default=0.1)
    
    parser.add_argument('--n2g', type=str2bool, default = True)
    parser.add_argument('--g2g', type=str2bool, default = True)
    parser.add_argument('--sed_sim', type=str2bool, default = True)
    parser.add_argument('--iou_sim', type=str2bool, default = True)
    parser.add_argument('--multi_scale', type=str2bool, default = False)
    parser.add_argument('--multi_scale_dyn', type=str2bool, default = False)
    parser.add_argument('--init_hidden', type=str2bool, default = False)
    parser.add_argument('--fuse_type', type=str, default = 'GRU')
    parser.add_argument('--histo_dim', type=int, default = 10)
    parser.add_argument('--subgraph_pool', type=str, default = 'max')
    parser.add_argument('--graph_pool', type=str, default = 'max')
    parser.add_argument('--n2g_pool', type=str, default = 'max')
    parser.add_argument('--g2g_pool', type=str, default = 'max')
    parser.add_argument('--use_hidden', type=str2bool, default = True)
    parser.add_argument('--predict_method', type=str, default = 'mul')
    parser.add_argument('--weighted_loss', type=str2bool, default = False)
    
    args = parser.parse_args()
    run(args)
