import os
import time
import json
import random
import argparse
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

# ==== Import your modules (adjust paths as needed) ====
from infer_batch import id_mapping, feats, graph, labels, split_idx, model, data_loader
from losses import compute_cluster_loss_optimized, compute_user_level_contrast_loss
from sampling import sample_and_expand_users


def compute_sim_dissim_loss(
    embeddings: torch.Tensor,
    user_ids_batch: list,
    seed_users: list,
    U_b_similar: dict,
    U_b_dissimilar: dict,
    tau: float = 0.07,
    use_inbatch_negatives: bool = False,
    verbose: bool = False
) -> (torch.Tensor, list, list):
    """
    InfoNCE-style contrastive loss at the user level, with optional in-batch negatives.
    Returns:
        avg_loss: scalar tensor
        sim_pos_list: list of positive cosine similarities
        sim_neg_list: list of negative cosine similarities
    """
    device = embeddings.device
    embeddings = F.normalize(embeddings, dim=-1)
    user2idx = {uid: i for i, uid in enumerate(user_ids_batch)}

    total_loss = torch.tensor(0.0, device=device)
    count = 0
    sim_pos_list = []
    sim_neg_list = []

    for u in seed_users:
        if u not in user2idx:
            continue
        idx_u = user2idx[u]
        z_u = embeddings[idx_u].unsqueeze(0)  # [1, d]

        # positive indices
        pos_idxs = [user2idx[v] for v in U_b_similar.get(u, []) if v in user2idx]
        if not pos_idxs:
            continue

        # negative indices
        if use_inbatch_negatives:
            neg_idxs = [i for i, uid in enumerate(user_ids_batch) if i != idx_u]
        else:
            neg_idxs = [user2idx[n] for n in U_b_dissimilar.get(u, []) if n in user2idx]
        if not neg_idxs:
            continue

        Z_neg = embeddings[neg_idxs]            # [N_neg, d]
        sim_neg = F.cosine_similarity(z_u, Z_neg, dim=-1)  # [N_neg]
        sim_neg_list += sim_neg.detach().cpu().tolist()
        sim_neg_scaled = sim_neg / tau

        for idx_pos in pos_idxs:
            z_pos = embeddings[idx_pos].unsqueeze(0)  # [1, d]
            sim_pos = F.cosine_similarity(z_u, z_pos, dim=-1)[0]  # scalar
            sim_pos_list.append(sim_pos.detach().cpu().item())

            sim_pos_scaled = sim_pos / tau
            num = torch.exp(sim_pos_scaled)
            denom = num + torch.exp(sim_neg_scaled).sum()
            loss_uv = -torch.log(num / (denom + 1e-8))

            total_loss += loss_uv
            count += 1

            if verbose:
                print(f"[u={u}] sim_pos={sim_pos.item():.4f}, "
                      f"sim_neg_mean={sim_neg.mean().item():.4f}, "
                      f"loss_uv={loss_uv.item():.4f}, neg_count={len(sim_neg)}")

    if count == 0:
        return torch.tensor(0.0, device=device), sim_pos_list, sim_neg_list

    avg_loss = total_loss / count
    if verbose:
        print(f"Total sim_dissim pairs: {count}, Avg Loss: {avg_loss.item():.4f}")

    return avg_loss, sim_pos_list, sim_neg_list


def run_kmeans(embeddings: torch.Tensor, num_clusters: int = 10) -> np.ndarray:
    embeddings_np = embeddings.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embeddings_np)
    return kmeans.labels_

def load_user_dicts(user_map_path, similar_path, dissimilar_path):
    with open(user_map_path, "r") as f:
        user_map = json.load(f)
    with open(similar_path, "r") as f:
        similar_users_dict = json.load(f)
    with open(dissimilar_path, "r") as f:
        dissimilar_users_dict = json.load(f)
    return user_map, similar_users_dict, dissimilar_users_dict

def train_contrastive_clustering(
    model, feats, graph,
    user_map_path, similar_path, dissimilar_path,
    user_map, similar_users_dict, dissimilar_users_dict,
    device="cuda", epochs=10, seed_batch_size=1000,
    k=5, tau=0.07,
    lambda_cluster=1.0, lambda_sim=10.0, lambda_user=1.0,
    K=10, lr=1e-3
):
    # Setup
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    loss_records = []
    best_loss = float("inf")
    checkpoint_path = "workspace/baseline/GraphMae2/our_checkpoints/best_model.pt"

    for epoch in tqdm(range(1, epochs + 1), desc="Training Epochs"):
        # 1) Sample users
        U_b, U_b_expanded, U_b_similar, U_b_dissimilar = sample_and_expand_users(
            user_map_path, similar_path, dissimilar_path,
            seed_batch_size=seed_batch_size, k=k, random_seed=42 + epoch,
            user_map=user_map, similar_users_dict=similar_users_dict,
            dissimilar_users_dict=dissimilar_users_dict
        )

        # 2) Map to post node indices
        expanded_node_indices = set()
        for u in U_b_expanded + sum(U_b_similar.values(), []) + sum(U_b_dissimilar.values(), []):
            posts = user_map.get(u, [])
            if isinstance(posts, list): expanded_node_indices.update(posts)
            elif posts:       expanded_node_indices.add(posts)
        expanded_node_indices_int = [id_mapping[str(x)] for x in expanded_node_indices if str(x) in id_mapping]
        expanded_node_indices_int = expanded_node_indices_int[:20000]

        # 3) Collect embeddings
        all_embs, node_indices = [], []
        for batch in data_loader:
            batch_g, targets, lbls, node_idx = batch
            node_idx = node_idx.to(device)
            mask = torch.tensor([(i in expanded_node_indices_int) for i in node_idx[targets].tolist()], device=device)
            if not mask.any(): continue
            batch_g = batch_g.to(device)
            x = batch_g.ndata.pop("feat")
            with autocast():
                pred = model(batch_g, x)
            embs = pred[targets][mask]
            all_embs.append(embs)
            node_indices.extend(node_idx[targets][mask].tolist())
            del pred, x, batch_g, lbls, mask
            torch.cuda.empty_cache()
        expanded_embeddings = torch.cat(all_embs, dim=0)

        # 4) Build user_ids_batch
        inverse_map = {v: k for k, v in id_mapping.items()}
        post_to_user = {p: u for u, posts in user_map.items() for p in (posts if isinstance(posts,list) else [posts])}
        user_ids_batch = []
        valid_embs = []
        for nid, emb in zip(node_indices, expanded_embeddings):
            post_id = inverse_map.get(nid)
            uid = post_to_user.get(post_id)
            if uid:
                user_ids_batch.append(uid)
                valid_embs.append(emb)
        expanded_embeddings = torch.stack(valid_embs)

        # 5) Compute losses
        verbose = (epoch == 1)
        L_sim, pos_list, neg_list = compute_sim_dissim_loss(
            expanded_embeddings, user_ids_batch, U_b,
            U_b_similar, U_b_dissimilar,
            tau=tau, use_inbatch_negatives=True, verbose=verbose
        )
        L_cluster = compute_cluster_loss_optimized(expanded_embeddings, K=K)
        L_user    = compute_user_level_contrast_loss(expanded_embeddings, user_ids_batch, tau=tau)

        total_loss = lambda_sim * L_sim + lambda_cluster * L_cluster + lambda_user * L_user
        loss_records.append((epoch, L_sim.item(), L_cluster.item(), L_user.item(), total_loss.item()))

        # 6) Backward
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # 7) Logging & saving
        print(f"[Epoch {epoch}] L_sim={L_sim:.4f}, L_cluster={L_cluster:.4f}, "
              f"L_user={L_user:.4f}, total={total_loss:.4f}")
        # Loss curve
        os.makedirs("plots_sim", exist_ok=True)
        rec = np.array(loss_records)
        plt.figure()
        plt.plot(rec[:,0], rec[:,-1], label='Total Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(f'Loss Curve up to Epoch {epoch}')
        plt.grid(True); plt.legend()
        plt.savefig(f"plots_sim/loss_epoch_{epoch}.png"); plt.close()

        # Save pos/neg hist on first epoch
        if epoch == 1 and verbose:
            plt.figure(figsize=(8,4))
            plt.hist(pos_list, bins=50, alpha=0.6, label='pos')
            plt.hist(neg_list, bins=50, alpha=0.6, label='neg')
            plt.legend(); plt.title('Positive vs Negative Cosine Similarities')
            plt.savefig("plots_sim/pos_neg_similarity.png"); plt.close()

        # Checkpoint
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': best_loss
            }, checkpoint_path)
            print(f"Saved new best model at epoch {epoch} (loss={best_loss:.4f})")

    print("Training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--tau', type=float, default=0.07)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    user_map_path   = 'workspace/data/pretrain/user_map_combined.json'
    similar_path    = 'workspace/data/pretrain/similar_users_rbf_pos0_new2.json'
    dissimilar_path = 'workspace/data/pretrain/dissimilar_users_rbf_neg0_new2.json'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    user_map, sim_dict, dissim_dict = load_user_dicts(user_map_path, similar_path, dissimilar_path)

    train_contrastive_clustering(
        model=model,
        feats=feats,
        graph=graph,
        user_map_path=user_map_path,
        similar_path=similar_path,
        dissimilar_path=dissimilar_path,
        user_map=user_map,
        similar_users_dict=sim_dict,
        dissimilar_users_dict=dissim_dict,
        device=device,
        epochs=30,
        seed_batch_size=2000,
        tau=0.05,
        lr=1e-3,
    )
