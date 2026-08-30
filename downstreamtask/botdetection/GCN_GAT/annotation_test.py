import pandas as pd
import json
import torch
import os.path as osp
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from model import BotGAT, BotRGCN
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hidden_dim = 128
batch_size = 1024
lr = 1e-3
weight_decay = 1e-5
dropout = 0.3
no_up_limit = 4
max_epoch=1000

def metrics(truth, preds):
    return accuracy_score(truth, preds), \
           f1_score(truth, preds), \
           precision_score(truth, preds), \
           recall_score(truth, preds)


def train_one_epoch():
    model.train()
    pbar = tqdm(train_loader, ncols=0)
    for batch in pbar:
        optimizer.zero_grad()
        batch = batch.to(device)
        out = model(batch.des_embedding,
                    batch.tweet_embedding,
                    batch.num_property_embedding,
                    batch.cat_property_embedding,
                    batch.edge_index,
                    batch.edge_type)
        out = out[:batch.batch_size]
        label = batch.y[:batch.batch_size]
        loss = loss_fn(out, label)
        loss.backward()
        optimizer.step()
        pbar.set_postfix_str('{:.6f}'.format(loss))


@torch.no_grad()
def validation(loader):
    model.eval()
    all_truth = []
    all_preds = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.des_embedding,
                    batch.tweet_embedding,
                    batch.num_property_embedding,
                    batch.cat_property_embedding,
                    batch.edge_index,
                    batch.edge_type)
        out = out[:batch.batch_size]
        label = batch.y[:batch.batch_size]
        all_truth.append(label.to('cpu'))
        all_preds.append(out.argmax(dim=-1).to('cpu'))
    all_truth = torch.cat(all_truth).numpy()
    all_preds = torch.cat(all_preds).numpy()
    return metrics(all_truth, all_preds)


if __name__ == '__main__':
    # 1) Load split and label CSVs
    split_df = pd.read_csv('workspace/workspace/TwiBot-22/datasets/Twibot-22/split.csv')
    label_df = pd.read_csv('workspace/workspace/TwiBot-22/datasets/Twibot-22/label.csv')

    # 2) Build id->node_idx mapping
    author_df = pd.read_csv(
        'workshop/workspace/data/finetune/twibot/author_ids.csv',
        dtype=str
    )
    idx_map = {aid.lstrip('u'): i for i, aid in enumerate(author_df['author_id'].tolist())}

    # 3) Build labels tensor
    tmp = [0] * len(label_df)
    for item in tqdm(label_df.itertuples(), total=len(label_df), desc="Building labels"):
        raw_id = item[1].lstrip('u')
        if raw_id not in idx_map:
            continue
        node_i = idx_map[raw_id]
        tmp[node_i] = 1 if item[2] == 'bot' else 0
    labels = torch.tensor(tmp, dtype=torch.long)

    # 4) Build train/val/test masks
    train_idx, val_idx, test_idx = [], [], []
    for item in tqdm(split_df.itertuples(), total=len(split_df), desc="Building masks"):
        raw_id = item[1].lstrip('u')
        if raw_id not in idx_map:
            continue
        node_i = idx_map[raw_id]
        if item[2] == 'train':
            train_idx.append(node_i)
        elif item[2] == 'val':
            val_idx.append(node_i)
        elif item[2] == 'test':
            test_idx.append(node_i)

    print(f"Masks sizes — train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

    train_mask = torch.tensor(train_idx, dtype=torch.long)
    val_mask   = torch.tensor(val_idx,   dtype=torch.long)
    test_mask  = torch.tensor(test_idx,  dtype=torch.long)

    # 5) Load embeddings and graph
    path = 'workshop/workspace/workspace/TwiBot-22/src/BotRGCN/twibot_22/processed_data'
    des_emb   = torch.load(os.path.join(path, 'des_tensor.pt'))
    tweet_emb = torch.load(os.path.join(path, 'tweets_tensor.pt'))
    num_emb   = torch.load(os.path.join(path, 'num_properties_tensor.pt'))
    cat_emb   = torch.load(os.path.join(path, 'cat_properties_tensor.pt'))
    edge_index= torch.load(os.path.join(path, 'edge_index.pt')).contiguous()
    edge_type = torch.load(os.path.join(path, 'edge_type.pt')).contiguous()

    data = Data(
        edge_index=edge_index,
        edge_type=edge_type,
        y=labels,
        des_embedding=des_emb,
        tweet_embedding=tweet_emb,
        num_property_embedding=num_emb,
        cat_property_embedding=cat_emb,
        num_nodes=labels.shape[0]
    )

    # 6) Create loaders
    train_loader = NeighborLoader(data, num_neighbors=[256]*4, batch_size=batch_size, input_nodes=train_mask, shuffle=True)
    val_loader   = NeighborLoader(data, num_neighbors=[256]*4, batch_size=batch_size, input_nodes=val_mask)
    test_loader  = NeighborLoader(data, num_neighbors=[256]*4, batch_size=batch_size, input_nodes=test_mask)

    # 7) Initialize model and optimizer
    model     = BotGAT(hidden_dim=hidden_dim, dropout=dropout, num_prop_size=num_emb.shape[-1], cat_prop_size=cat_emb.shape[-1]).to(device)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 8) Training with early stopping
    best_state, best_acc, no_up = model.state_dict(), 0, 0
    for epoch in range(max_epoch):
        print(f"Epoch {epoch+1}/{max_epoch}")
        train_one_epoch()
        val_metrics = validation(val_loader)
        print(f"  Val Acc {val_metrics[0]:.4f} F1 {val_metrics[1]:.4f} No-up {no_up}")
        if val_metrics[0] > best_acc:
            best_acc, best_state, no_up = val_metrics[0], model.state_dict(), 0
        else:
            no_up += 1
            if no_up >= no_up_limit:
                print("Early stopping!")
                break

    # 9) Test evaluation
    model.load_state_dict(best_state)
    tm = validation(test_loader)
    print("\n=== Test Metrics ===")
    print(f"test : acc {tm[0]:.6f}  f1 {tm[1]:.6f}  pre {tm[2]:.6f}  rec {tm[3]:.6f}")