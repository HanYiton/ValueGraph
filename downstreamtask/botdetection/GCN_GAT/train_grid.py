#!/usr/bin/env python3
import os
import csv
import itertools
import torch
import torch.nn as nn
from argparse import ArgumentParser
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from utils import null_metrics, calc_metrics, is_better
from dataset import get_train_data
from model import BotGAT, BotGCN, BotRGCN

import random
import numpy as np
seed = 42
# 0、1、2、42、100
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset',     type=str,   default='Twibot-22')
    parser.add_argument('--mode',        type=str,   default='RGCN')
    parser.add_argument('--max_epoch',   type=int,   default=1000)
    parser.add_argument('--no_up',       type=int,   default=30)
    parser.add_argument('--output',      type=str,   default='results6.csv')
    return parser.parse_args()

def build_model(mode, hidden_dim, dropout, data):
    if mode == 'GAT':
        return BotGAT(hidden_dim=hidden_dim,
                      dropout=dropout,
                      num_prop_size=data.num_property_embedding.shape[-1],
                      cat_prop_size=data.cat_property_embedding.shape[-1])
    elif mode == 'GCN':
        return BotGCN(hidden_dim=hidden_dim,
                      dropout=dropout,
                      num_prop_size=data.num_property_embedding.shape[-1],
                      cat_prop_size=data.cat_property_embedding.shape[-1])
    elif mode == 'RGCN':
        return BotRGCN(hidden_dim=hidden_dim,
                       dropout=dropout,
                       num_prop_size=data.num_property_embedding.shape[-1],
                       cat_prop_size=data.cat_property_embedding.shape[-1],
                       num_relations=data.edge_type.max().item()+1)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def get_loaders(data, batch_size):
    return (
        NeighborLoader(data, num_neighbors=[256]*4, batch_size=batch_size,
                       input_nodes=data.train_idx, shuffle=True),
        NeighborLoader(data, num_neighbors=[256]*4, batch_size=batch_size,
                       input_nodes=data.val_idx, shuffle=False),
        NeighborLoader(data, num_neighbors=[256]*4, batch_size=batch_size,
                       input_nodes=data.test_idx, shuffle=False),
    )

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_label, all_logits = [], []
    loss_fn = nn.CrossEntropyLoss()
    total_loss, cnt = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        n = batch.batch_size
        out = model(batch.des_embedding,
                    batch.tweet_embedding,
                    batch.num_property_embedding,
                    batch.cat_property_embedding,
                    batch.edge_index,
                    batch.edge_type)[:n]
        label = batch.y[:n]
        all_label += label.cpu().tolist()
        all_logits += out.cpu().tolist()
        loss = loss_fn(out, label)
        total_loss += loss.item() * n
        cnt += n
    metrics, _ = calc_metrics(torch.tensor(all_label), torch.tensor(all_logits))
    return metrics

def train_and_eval(args, hyperparams):
    # unpack
    hd, lr, wd, dp, bs = (hyperparams[k] for k in ['hidden_dim','lr','weight_decay','dropout','batch_size'])
    # prepare data & model
    data = get_train_data(args.dataset)
    data.edge_index = data.edge_index.contiguous()
    data.edge_type  = data.edge_type.contiguous()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args.mode, hd, dp, data).to(device)
    train_loader, val_loader, test_loader = get_loaders(data, bs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn   = nn.CrossEntropyLoss()
    best_val = null_metrics()
    best_state = None
    early_stop = 0

    # training loop
    for epoch in range(args.max_epoch):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            n = batch.batch_size
            out = model(batch.des_embedding,
                        batch.tweet_embedding,
                        batch.num_property_embedding,
                        batch.cat_property_embedding,
                        batch.edge_index,
                        batch.edge_type)[:n]
            label = batch.y[:n]
            loss_fn(out, label).backward()
            optimizer.step()

        # validation
        val_metrics = evaluate(model, val_loader, device)
        if best_state is None or is_better(val_metrics, best_val):
            best_val = val_metrics
            best_state = model.state_dict()
            early_stop = 0
        else:
            early_stop += 1
        if early_stop >= args.no_up:
            break

    # test
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    return test_metrics

def main():
    args = parse_args()

    # define your search grid
    grid = {
        'hidden_dim':    [96,128, 256],
        'lr':            [5e-4, 2e-4],
        'weight_decay':  [1e-3, 1e-4,5e-5],
        'dropout':       [0.1,0.3, 0.5],
        'batch_size':    [128, 256],
    }

    # open CSV and write header
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        header = list(grid.keys()) + ['acc','f1-score','precision','recall','mcc','roc-auc','pr-auc']
        writer.writerow(header)

        # iterate over all combinations
        for values in itertools.product(*grid.values()):
            params = dict(zip(grid.keys(), values))
            print(f"Running with {params} ...")
            metrics = train_and_eval(args, params)
            row = values + tuple(metrics[k] for k in ['acc','f1-score','precision','recall','mcc','roc-auc','pr-auc'])
            writer.writerow(row)
            print(f" → Test acc={metrics['acc']:.4f}, f1={metrics['f1-score']:.4f}")

    print(f"Grid search done. Results saved to {args.output}")

if __name__ == '__main__':
    main()
