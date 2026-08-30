import torch
from datasets.lc_sampler import (
    load_dataset,
    setup_finetune_dataloder
)
import argparse
import importlib
import utils
importlib.reload(utils)

from utils import build_args_new
from models import build_model
from tqdm import tqdm

# Note: conda activate graphmae
# load data and mode;
device = "cuda"
# data_dir = '/workspace/data/finetune'
data_dir = '/workspace/data/pretrain'
pretrain_path = 'checkpoints/gat_gat_512_2_social_0.5_512_checkpoint.pt'
dataset_name = 'twitter'

feats, graph, labels, split_idx, id_mapping = load_dataset(
    data_dir, dataset_name, create_ego_graphs=False
)

# load model
args = build_args_new() 
args.num_features = feats.shape[1]
model = build_model(args)

model.load_state_dict(torch.load(pretrain_path))
model = model.get_encoder()
model.to(device)
model.eval()

ego_graphs_file_path = f"lc_ego_graphs/{dataset_name}/graphs.pt"
ego_graph_nodes = torch.load(ego_graphs_file_path, weights_only=False)
train_egs, val_egs, test_egs = ego_graph_nodes

### === Get the embeddings from here ===
# model.train()
batch_size = 16
print(batch_size)
data_loader = setup_finetune_dataloder(
    "lc", graph, feats, train_egs, labels, batch_size=batch_size, shuffle=False
)

### === this is only for inference not weight update ===
# epoch_iter = tqdm(data_loader)
# lis = []
# with torch.no_grad():
#     for batch in epoch_iter:
#         batch_g, targets, batch_lbls, node_idx = batch
#         batch_g = batch_g.to(device)
#         x = batch_g.ndata.pop("feat")
#         prediction = model(batch_g, x)
#         batch_emb = prediction[targets]
#         lis.append(batch_emb.cpu())

# out = torch.cat(lis, dim=0)
# print(out.shape)


# === code to run in training ===

# epoch_iter = tqdm(data_loader)
# lis = []
# node_indices = []  
# target_nodes = [10, 100, 200, 5000, 10000, 50000, 100000, 500000, 1000000]

# for batch in epoch_iter:
#     batch_g, targets, batch_lbls, node_idx = batch
    
#     # Convert node_idx to set for faster lookup
#     batch_target_mask = torch.tensor([idx.item() in target_nodes for idx in node_idx[targets]])
    
#     # Skip if no target nodes in this batch
#     if not batch_target_mask.any():
#         continue
        
#     # Process only if we have target nodes in this batch
#     batch_g = batch_g.to(device)
#     x = batch_g.ndata.pop("feat")
#     prediction = model(batch_g, x)
#     batch_emb = prediction[targets][batch_target_mask]

#     # Store embeddings and corresponding node indices
#     lis.append(batch_emb)
#     node_indices.extend(node_idx[targets][batch_target_mask].tolist())

#     del prediction, x, batch_g, batch_lbls, batch_target_mask
#     torch.cuda.empty_cache()

# del lis
# full_embeddings = torch.cat(lis, dim=0)
# print(len(node_indices))
# print(full_embeddings.shape)

