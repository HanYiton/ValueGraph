import torch
from datasets.lc_sampler import (
    load_dataset
)
import argparse
from utils import build_args
from models import build_model

# Note: conda activate graphmae
# load data and mode;
device = "cuda"
data_dir = '/workspace/data/finetune'
# data_dir = '/workspace/data/pretrain'
pretrain_path = 'checkpoints/gat_gat_512_2_social_0.5_512_checkpoint.pt'
# dataset_name = 'twitter'
dataset_name = 'twibot'
feats, graph, labels, split_idx, id_mapping = load_dataset(
    data_dir, dataset_name, create_ego_graphs=False
)
print(f"features size : {feats.shape[1]}")

# get default args
args = build_args()
args.num_features = feats.shape[1]
eval_model = build_model(args)
eval_model.load_state_dict(torch.load(pretrain_path))
eval_model = eval_model.get_encoder()
eval_model.to(device)
# eval_model.train()
eval_model.eval()

with torch.no_grad():
    out = eval_model(graph.to(device), feats.to(device)).cpu().numpy()
    print()

import os
import numpy as np
temp = np.load("/workspace/data/finetune/twibot/embeddings/twibot_feat_1.npy")
out = np.vstack(
    [
        temp,
        out,
    ]
)
print(len(out))
np.save("/workspace/data/finetune/twibot/embeddings/twibot_feat_1", out)
# del temp, out

# print('validate', np.load("/workspace/data/finetune/twibot/embeddings/twibot_feat.npy").shape)
# import ipdb
# ipdb.set_trace()

# shape [num_nodes, 512]
# for i in tqdm(range(0, feats.shape[0], batch_size)):
#     batch_feats = feats[i:i + batch_size]
#     batch_out = eval_model(graph, batch_feats) 
#     outputs.append(batch_out) 
# out = torch.cat(outputs, dim=0)

# import ipdb; ipdb.set_trace()

# out = eval_model(graph.to(device), feats.to(device))
# print(f'Shape: {out.shape}')