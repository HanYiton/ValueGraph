import torch
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

# torch.cuda.set_device(1)

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
    model, data_loader, id_mapping, user_map, similar_users_dict, dissimilar_users_dict,
    device="cuda", epochs=3, seed_batch_size=10, k=5, tau=0.07,
    lambda_cluster=1.0, K=10, lr=1e-3
):
    """
    Contrastive + clustering training loop with GPU acceleration and plotting.
    """
    model = model.to(device)
    scaler = GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")
    loss_records = []
    checkpoint_path = "/workspace/baseline/GraphMae2/our_checkpoints/best_model.pt"

    for epoch in range(1, epochs + 1):
        # 1) Sample users
        U_b, U_b_expanded, U_b_similar, U_b_dissimilar = sample_and_expand_users(
            user_map, similar_users_dict, dissimilar_users_dict,
            seed_batch_size=seed_batch_size, k=k, random_seed=42 + epoch
        )

        # 2) Build expanded_node_idxs
        expanded_post_ids = set(U_b_expanded)
        for sim in U_b_similar.values():
            expanded_post_ids.update(sim)
        for dis in U_b_dissimilar.values():
            expanded_post_ids.update(dis)

        expanded_node_idxs = [
            id_mapping[str(pid)]
            for pid in expanded_post_ids
            if str(pid) in id_mapping
        ]
        print(f"[Epoch {epoch}] expanded_node_idxs sample:", expanded_node_idxs[:5], "total:", len(expanded_node_idxs))
        expanded_node_idxs = expanded_node_idxs[:20_000]

        # 3) Forward: collect embeddings
        model.train()
        all_embs = []
        all_nodes = []
        for batch_g, targets, _, node_idx in tqdm(data_loader, desc=f"Epoch {epoch} batches"):
            batch_g = batch_g.to(device, non_blocking=True)
            x = batch_g.ndata.pop("feat").to(device, non_blocking=True)

            with autocast():
                preds = model(batch_g, x)  # [num_nodes, feat_dim]

            # mask to only our sampled nodes
            mask = torch.tensor(
                [nid.item() in expanded_node_idxs for nid in node_idx[targets]],
                device=device
            )
            if not mask.any():
                continue

            emb = preds[targets][mask]
            all_embs.append(emb)
            all_nodes.extend(node_idx[targets][mask].tolist())

        # 🛡️ Empty check
        if not all_embs:
            print(f"[Epoch {epoch}] Warning: no target embeddings collected, skipping epoch.")
            continue

        expanded_embeddings = torch.cat(all_embs, dim=0)  # [N_samples, feat_dim]

        # 4) Map embeddings back to user IDs
        inverse_map = {v: k for k, v in id_mapping.items()}
        post_to_user = {pid: u for u, posts in user_map.items() for pid in posts}

        user_ids_batch = []
        valid_embs = []
        for emb, node in zip(expanded_embeddings, all_nodes):
            pid = inverse_map.get(node)
            u = post_to_user.get(pid)
            if u is not None:
                user_ids_batch.append(u)
                valid_embs.append(emb)
        expanded_embeddings = torch.stack(valid_embs).to(device)

        print("Top user frequencies:", Counter(user_ids_batch).most_common(5))

        # 5) Compute losses under AMP
        with autocast():
            L_user = compute_user_level_contrast_loss(
                expanded_embeddings, user_ids_batch, tau=tau
            )
            L_sim = compute_sim_dissim_loss(
                expanded_embeddings, user_ids_batch,
                U_b, U_b_similar, U_b_dissimilar,
                tau=tau
            )
        # clustering on CPU
        cluster_labels = run_kmeans(expanded_embeddings, num_clusters=K)
        L_cls = compute_cluster_loss_optimized(
            expanded_embeddings, cluster_labels, alpha=1.0
        )
        total_loss = L_user + L_sim + lambda_cluster * L_cls

        # 6) Backprop + optimizer step
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 7) Record & plot losses
        loss_records.append([
            epoch, L_user.item(), L_sim.item(), L_cls.item(), total_loss.item()
        ])
        arr = np.array(loss_records)
        plt.figure()
        plt.plot(arr[:, 0], arr[:, 1], label='L_user')
        plt.plot(arr[:, 0], arr[:, 2], label='L_sim')
        plt.plot(arr[:, 0], arr[:, 3], label='L_cluster')
        plt.plot(arr[:, 0], arr[:, 4], label='Total')
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title(f"Loss up to Epoch {epoch}")
        plt.legend(); plt.grid(True)
        os.makedirs("plots_new_1", exist_ok=True)
        plt.savefig(f"plots_new_1/loss_epoch_{epoch}.png")
        plt.close()

        # 8) Save best model
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss
            }, checkpoint_path)
            print(f"Saved new best model at epoch {epoch}, loss {best_loss:.4f}")

        # 9) Log GPU memory
        print(f"[Epoch {epoch}] Total={total_loss.item():.4f} | GPU Mem: {torch.cuda.memory_reserved(device)/1024**2:.1f} MB")

    return model

#################################################
# Usage Example
#################################################
if __name__ == "__main__":
    # user_map_path = "/workspace/data/pretrain/twitter/user_map.json"
    # similar_path = "/workspace/data/pretrain/twitter/similar_users_rbf_full_90.json"
    # dissimilar_path = "/workspace/data/pretrain/twitter/dissimilar_users_rbf_full_90.json"

    user_map_path = '/workspace/data/pretrain/user_map_combined.json'
    similar_path = "/workspace/data/pretrain/similar_users_rbf_full_90_new.json"
    dissimilar_path = "/workspace/data/pretrain/dissimilar_users_rbf_full_90_new.json"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    user_map, similar_users_dict, dissimilar_users_dict = load_user_dicts(
        user_map_path, similar_path, dissimilar_path
    )
    train_contrastive_clustering1(
        model=model,
        data_loader=data_loader,
        id_mapping=id_mapping,
        user_map=user_map,
        similar_users_dict=similar_users_dict,
        dissimilar_users_dict=dissimilar_users_dict,
        device=device,
        epochs=30,
        seed_batch_size=1000,
        k=5,
        tau=0.1,
        lambda_cluster=0.1,
        K=3,
        lr=1e-3
    )
    
