import torch
torch.cuda.set_device(0)
import torch.nn.functional as F
import torch.nn as nn
import random
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from collections import defaultdict
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from collections import Counter
import os
import time
import matplotlib.pyplot as plt
# from infer import out, id_mapping, feats, graph, labels, split_idx, eval_model  # 'out' is shape [num_nodes, 512]
# from infer import id_mapping, feats, graph, labels, split_idx, eval_model  # 'out' is shape [num_nodes, 512]
from infer_batch import id_mapping, feats, graph, labels, split_idx, model, data_loader
from losses import compute_user_level_contrast_loss, compute_sim_dissim_loss, compute_cluster_loss_optimized
from sampling import sample_and_expand_users

from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader



def load_user_dicts(user_map_path, similar_path, dissimilar_path):
    with open(user_map_path, "r") as f:
        user_map = json.load(f)
    with open(similar_path, "r") as f:
        similar_users_dict = json.load(f)
    with open(dissimilar_path, "r") as f:
        dissimilar_users_dict = json.load(f)
    return user_map, similar_users_dict, dissimilar_users_dict

def run_kmeans(embeddings, num_clusters=10):
    embeddings_np = embeddings.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embeddings_np)
    return kmeans.labels_

def train_contrastive_clustering1(
    model, feats, graph, user_map_path, similar_path, dissimilar_path,
    user_map, similar_users_dict, dissimilar_users_dict,
    device="cuda", epochs=3, seed_batch_size=10, k=5, tau=0.07,
    lambda_cluster=1.0,lambda_sim=100,lambda_user=0.5, K=10, kmeans_update_freq=1, lr=1e-4
):
    loss_records = []
    
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    best_loss = float("inf")
    checkpoint_path = "workspace/baseline/GraphMae2/our_checkpoints/best_model.pt"
    
    for epoch in tqdm(range(1, epochs + 1)):
        U_b, U_b_expanded, U_b_similar, U_b_dissimilar = sample_and_expand_users(
            user_map_path, similar_path, dissimilar_path,
            seed_batch_size=seed_batch_size, k=k, random_seed=42+epoch,
            user_map=user_map, similar_users_dict=similar_users_dict,
            dissimilar_users_dict=dissimilar_users_dict
        )

        # Ensure similar/dissimilar users are mapped to nodes
        expanded_node_indices = []
        for u in U_b_expanded:
            val = user_map.get(u, [])
            if isinstance(val, list):
                expanded_node_indices.extend(val)
            elif val:
                expanded_node_indices.append(val)
        for sim_list in U_b_similar.values():
            for u in sim_list:
                val = user_map.get(u, [])
                if isinstance(val, list):
                    expanded_node_indices.extend(val)
                elif val:
                    expanded_node_indices.append(val)
        for dissim_list in U_b_dissimilar.values():
            for u in dissim_list:
                val = user_map.get(u, [])
                if isinstance(val, list):
                    expanded_node_indices.extend(val)
                elif val:
                    expanded_node_indices.append(val)
        expanded_node_indices = list(set(expanded_node_indices))
        expanded_node_indices_int = [id_mapping[str(x)] for x in expanded_node_indices if str(x) in id_mapping]

        # to prevent overloading
        expanded_node_indices_int = expanded_node_indices_int[:20_000]
        # import pdb; pdb.set_trace()
        lis = []
        node_indices = []  
        epoch_iter = tqdm(data_loader)
        for batch in epoch_iter:
            batch_g, targets, batch_lbls, node_idx = batch
            
            # Convert node_idx to set for faster lookup
            batch_target_mask = torch.tensor([idx.item() in expanded_node_indices_int for idx in node_idx[targets]])
            
            # Skip if no target nodes in this batch
            if not batch_target_mask.any():
                continue
                
            # Process only if we have target nodes in this batch
            batch_g = batch_g.to(device)
            x = batch_g.ndata.pop("feat")
            prediction = model(batch_g, x)
            batch_emb = prediction[targets][batch_target_mask]

            # Store embeddings and corresponding node indices
            lis.append(batch_emb)
            node_indices.extend(node_idx[targets][batch_target_mask].tolist())

            del prediction, x, batch_g, batch_lbls, batch_target_mask
            torch.cuda.empty_cache()
        # import pdb; pdb.set_trace()
        expanded_embeddings = torch.cat(lis, dim=0)
        del lis

        inverse_id_mapping = {v: k for k, v in id_mapping.items()}
        
        user_ids_batch = []
        valid_embeddings = []
        post_id_to_user = {}
        for u, post_ids in user_map.items():
            for p_id in post_ids:
                post_id_to_user[p_id] = u
        for i, node_id in enumerate(node_indices):
            post_id = inverse_id_mapping.get(node_id)  # get post_id from node_id
            if post_id is None:
                continue
            found_uid = post_id_to_user.get(post_id)
            if found_uid is None:
                continue
            user_ids_batch.append(found_uid)
            valid_embeddings.append(expanded_embeddings[i])
        expanded_embeddings = torch.stack(valid_embeddings)
        print("Top user frequencies:", Counter(user_ids_batch).most_common(5))
        # print(f"[Epoch {epoch}] valid user anchor count in L_user: {len(set([u for u in user_ids_batch if user_ids_batch.count(u) > 1]))}")

        # L_user = compute_user_level_contrast_loss(expanded_embeddings, user_ids_batch, tau=tau)
        L_sim_dissim = compute_sim_dissim_loss(
            expanded_embeddings, user_ids_batch, U_b, U_b_similar, U_b_dissimilar, tau=0.07
        )
        cluster_labels = run_kmeans(expanded_embeddings, num_clusters=K)
        # L_cluster = compute_cluster_loss(expanded_embeddings, cluster_labels, alpha=1.0)
    #     L_cluster = compute_cluster_loss_optimized(
    # expanded_embeddings,
    # cluster_labels,
    # margin=1.0,    # 你可以从 0.5~2.0 尝试
    # beta=0.5)       # 你可以从 0.1~1.0 尝试)

        # total_loss = lambda_user*L_user + lambda_sim*L_sim_dissim + lambda_cluster * L_cluster
        total_loss = L_sim_dissim
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        global_step += 1
        
        loss_records.append([
        epoch, 
        # L_user.item(), 
        L_sim_dissim.item(), 
        # L_cluster.item(), 

        total_loss.item()
    ])
        # print(
        #     f"[Epoch {epoch}] step={global_step}, "
        #     f"L_user={L_user.item():.4f}, L_sim_dissim={L_sim_dissim.item():.4f}, "
        #     f"L_cluster={L_cluster.item():.4f}, Total={total_loss.item():.4f}"
        # )
        os.makedirs("plots_sim", exist_ok=True)
        # os.makedirs("loss_data", exist_ok=True)
        loss_matrix = np.array(loss_records)
        # np.save(f"loss_data/loss_matrix_epoch{epoch}.npy", loss_matrix)
        plt.figure()
        # plt.plot(loss_matrix[:, 0], lambda_user*loss_matrix[:, 1], label='L_user')
        plt.plot(loss_matrix[:, 0], lambda_sim*loss_matrix[:, 2], label='L_sim_dissim')
        # plt.plot(loss_matrix[:, 0], lambda_cluster*loss_matrix[:, 3], label='L_cluster')
        # plt.plot(loss_matrix[:, 0], loss_matrix[:, 4], label='Total Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"Loss Curves up to Epoch {epoch}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"plots_sim/loss_epoch_{epoch}.png")
        plt.close()
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'global_step': global_step
            }, checkpoint_path)
            print(f"Saved new best model at step {global_step} with loss {best_loss:.4f}")

        print(f"Cuda Usage: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
        print(f'Deleting expanded embeddings to clear cuda')
        del expanded_embeddings  
        torch.cuda.empty_cache()
        print(f"Cuda Usage After clearing: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
        

#################################################
# Usage Example
#################################################
if __name__ == "__main__":
    # user_map_path = "/workspace/data/pretrain/twitter/user_map.json"
    # similar_path = "/workspace/data/pretrain/twitter/similar_users_rbf_full_90.json"
    # dissimilar_path = "/workspace/data/pretrain/twitter/dissimilar_users_rbf_full_90.json"

    user_map_path = 'workspace/data/pretrain/user_map_combined.json'
    similar_path = "workspace/data/pretrain/similar_users_rbf_full_90_new.json"
    dissimilar_path = "workspace/data/pretrain/dissimilar_users_rbf_full_90_new.json"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    user_map, similar_users_dict, dissimilar_users_dict = load_user_dicts(
        user_map_path, similar_path, dissimilar_path
    )
    train_contrastive_clustering1(
        model=model,
        feats=feats,  
        graph=graph,
        user_map_path=user_map_path,
        similar_path=similar_path,
        dissimilar_path=dissimilar_path,
        user_map=user_map,
        similar_users_dict=similar_users_dict, 
        dissimilar_users_dict=dissimilar_users_dict,
        device=device,
        epochs=20,
        seed_batch_size=1000,
        k=5,
        tau=0.1,
        lambda_cluster=4,
        lambda_sim=10,
        lambda_user=1,
        K=10,
        lr=5e-4
        # kmeans_update_freq=1,
    )  