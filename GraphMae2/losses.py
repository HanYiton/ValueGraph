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

import torch
import torch.nn.functional as F
from collections import defaultdict

def compute_user_level_contrast_loss(embeddings, user_ids, tau=0.07, chunk_size=1024):
    """
    InfoNCE-style loss for pulling together embeddings from the same user.

    Args:
        embeddings: (N, d) tensor of user/post embeddings
        user_ids:   list of length N with user ID for each embedding
        tau:        temperature for softmax
        chunk_size: chunk size to reduce memory usage

    Returns:
        Scalar tensor representing average contrastive loss
    """
    device = embeddings.device
    N = embeddings.shape[0]
    if N == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    if torch.is_tensor(user_ids):
        user_ids = user_ids.cpu().tolist()

    normed_embeddings = F.normalize(embeddings, dim=-1)
    user_map_index = defaultdict(list)
    for idx, uid in enumerate(user_ids):
        user_map_index[uid].append(idx)

    total_loss = 0.0
    valid_anchors = 0

    for i_start in range(0, N, chunk_size):
        i_end = min(N, i_start + chunk_size)
        anchor_batch = normed_embeddings[i_start:i_end]
        sims = anchor_batch @ normed_embeddings.T
        log_probs = F.log_softmax(sims / tau, dim=1)

        for local_idx, anchor_idx in enumerate(range(i_start, i_end)):
            uid = user_ids[anchor_idx]
            pos_indices = [p for p in user_map_index[uid] if p != anchor_idx]
            if not pos_indices:
                continue
            pos_log_prob_sum = log_probs[local_idx, pos_indices].sum()
            total_loss += pos_log_prob_sum
            valid_anchors += 1

    if valid_anchors == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return -(total_loss / valid_anchors)

#################################################
# Similarity and Dissimilary Loss
#################################################

# def compute_sim_dissim_loss(
#     embeddings,
#     user_ids_batch,
#     seed_users,
#     U_b_similar,
#     U_b_dissimilar,
#     margin=0.2
# ):
#     """
#     Enforce that each seed user u is closer to users in U_b_similar[u]
#     than to users in U_b_dissimilar[u], by at least 'margin':

#     L_sd = average over all (u, v, w) of max(0, margin - (cos(u,v) - cos(u,w))).

#     * embeddings:       Tensor [N, d]
#     * user_ids_batch:   list of length N, giving the user_id for each row in 'embeddings'
#     * seed_users:       the list U_b (seed)
#     * U_b_similar[u]:   list of similar user IDs for user 'u'
#     * U_b_dissimilar[u]: likewise dissimilar
#     * margin:           margin for ranking
#     """
#     device = embeddings.device
#     # Build quick lookup: user_id -> index in 'embeddings'
#     user2idx = {}
#     for i, uid in enumerate(user_ids_batch):
#         user2idx[uid] = i

#     total_cost = torch.tensor(0.0, device=device)
#     triplet_count = 0

#     for seed_u in seed_users:
#         if seed_u not in user2idx:
#             continue
#         u_idx = user2idx[seed_u]
#         z_u = embeddings[u_idx]  # shape [d]

#         # all similar users for this seed
#         sim_list = U_b_similar.get(seed_u, [])
#         # all dissimilar
#         dissim_list = U_b_dissimilar.get(seed_u, [])

#         for sim_u in sim_list:
#             if sim_u not in user2idx:
#                 continue
#             z_sim = embeddings[user2idx[sim_u]]
#             # cos(u, sim)
#             cos_sim = F.cosine_similarity(
#                 z_u.unsqueeze(0), z_sim.unsqueeze(0), dim=-1
#             )[0]

#             for dis_u in dissim_list:
#                 if dis_u not in user2idx:
#                     continue
#                 z_dis = embeddings[user2idx[dis_u]]
#                 # cos(u, dis)
#                 cos_dis = F.cosine_similarity(
#                     z_u.unsqueeze(0), z_dis.unsqueeze(0), dim=-1
#                 )[0]

#                 # margin-based loss: want cos(u, sim) - cos(u, dis) >= margin
#                 cost = torch.relu(margin - (cos_sim - cos_dis))
#                 total_cost += cost
#                 triplet_count += 1

#     if triplet_count == 0:
#         return torch.tensor(0.0, device=device)

#     return total_cost / triplet_count

# import torch.nn.functional as F

# def compute_sim_dissim_loss(
#     embeddings: torch.Tensor,
#     user_ids_batch: list,
#     seed_users: list,
#     U_b_similar: dict,
#     U_b_dissimilar: dict,
#     tau: float = 0.07,
#     verbose: bool = False
# ) -> torch.Tensor:
#     """
#     InfoNCE-style contrastive loss at the user level.
#     For each seed user u, and each positive v in U_b_similar[u],
#     we treat U_b_dissimilar[u] as negatives and compute:

#     ℓ(u, v) = -log [ exp(sim(u,v)/τ) / (exp(sim(u,v)/τ) + ∑_{w in D(u)} exp(sim(u,w)/τ)) ]

#     Args:
#         embeddings:        [N, d] tensor of user embeddings
#         user_ids_batch:    list of user IDs corresponding to rows of embeddings
#         seed_users:        list of user IDs to compute contrastive loss for
#         U_b_similar:       dict {u: [similar user IDs]}
#         U_b_dissimilar:    dict {u: [dissimilar user IDs]}
#         tau:               temperature
#         verbose:           whether to print debug info

#     Returns:
#         Scalar tensor representing average contrastive loss
#     """
#     device = embeddings.device
#     sim_pos_list=[]
#     sim_neg_list=[]

#     # L2 normalize embeddings for cosine similarity stability
#     embeddings = F.normalize(embeddings, dim=-1)

#     # map user ID to row index
#     user2idx = {uid: i for i, uid in enumerate(user_ids_batch)}

#     total_loss = torch.tensor(0.0, device=device)
#     count = 0

#     for u in seed_users:
#         if u not in user2idx:
#             continue
#         z_u = embeddings[user2idx[u]].unsqueeze(0)  # [1, d]

#         pos_list = U_b_similar.get(u, [])
#         neg_list = U_b_dissimilar.get(u, [])

#         neg_idxs = [user2idx[n] for n in neg_list if n in user2idx]
#         if len(neg_idxs) == 0:
#             continue

#         Z_neg = embeddings[neg_idxs]  # [N_neg, d]
#         sim_neg = F.cosine_similarity(z_u, Z_neg, dim=-1)  # [N_neg]

#         for v in pos_list:
#             if v not in user2idx:
#                 continue
#             z_pos = embeddings[user2idx[v]].unsqueeze(0)  # [1, d]
#             sim_pos = F.cosine_similarity(z_u, z_pos, dim=-1)[0]  # scalar
#             sim_pos.append(sim_pos_list)
#             sim_neg.append(sim_neg_list)
#             sim_pos_scaled = sim_pos / tau
#             sim_neg_scaled = sim_neg / tau

#             num = torch.exp(sim_pos_scaled)
#             denom = num + torch.exp(sim_neg_scaled).sum()

#             loss_uv = -torch.log(num / denom)
#             total_loss += loss_uv
#             count += 1

#             # Debugging information
#             if verbose:
#                 print(f"[u={u}, v={v}] sim_pos={sim_pos.item():.4f}, "
#                       f"sim_neg_mean={sim_neg.mean().item():.4f}, "
#                       f"loss_uv={loss_uv.item():.4f}, "
#                       f"neg_count={len(sim_neg)}")

#     if count == 0:
#         return torch.tensor(0.0, device=device)
    
#     avg_loss = total_loss / count
#     if verbose:
#         print(f"Total sim_dissim pairs: {count}, Avg Loss: {avg_loss.item():.4f}")
        
#         plt.hist(sim_pos_list, bins=50, alpha=0.6, label='pos')
#         plt.hist(sim_neg_list, bins=50, alpha=0.6, label='neg')
#         plt.legend(); plt.show()
#     return avg_loss


# def compute_cluster_loss_optimized(embeddings, cluster_labels, alpha=1.0):
#     """
#     Computes the same cluster-separation loss:
#         L = mean_intra - alpha * mean_inter
#     without constructing the full NxN matrix. This saves significant
#     compute/memory cost on large N.
#     """
#     device = embeddings.device
#     N = embeddings.shape[0]
#     if N == 0:
#         return torch.tensor(0.0, device=device)

#     # Ensure cluster_labels is on the same device and is long
#     if not torch.is_tensor(cluster_labels):
#         cluster_labels = torch.tensor(cluster_labels, dtype=torch.long, device=device)
#     else:
#         cluster_labels = cluster_labels.to(device=device, dtype=torch.long)

#     # --- Global sums for entire dataset ---
#     # Sum of squares of each row
#     row_sqnorms = (embeddings ** 2).sum(dim=1)  # shape [N]
#     sum_of_sqnorms = row_sqnorms.sum()         # scalar
#     # Sum of all embeddings (vector in R^d), then squared norm
#     sum_of_embeddings = embeddings.sum(dim=0)  # shape [d]
#     sum_of_embeddings_sq = (sum_of_embeddings ** 2).sum()  # scalar

#     # sum_total = sum_{i != j} || x_i - x_j ||^2
#     #           = 2 * [ N * sum_of_sqnorms - ||sum_of_embeddings||^2 ]
#     sum_total = 2.0 * (N * sum_of_sqnorms - sum_of_embeddings_sq)

#     # --- Per-cluster sums ---
#     # We'll group embeddings by cluster and track:
#     #   1) sum of embeddings (to get the centroid & norm)
#     #   2) sum of squared norms
#     #   3) count (size of cluster)
#     cluster_sums = {}
#     cluster_sqnorm_sums = {}
#     cluster_counts = {}

#     for x, c in zip(embeddings, cluster_labels):
#         if c not in cluster_sums:
#             cluster_sums[c] = torch.zeros_like(x)
#             cluster_sqnorm_sums[c] = torch.tensor(0.0, device=device)
#             cluster_counts[c] = 0
#         cluster_sums[c] += x
#         cluster_sqnorm_sums[c] += (x * x).sum()
#         cluster_counts[c] += 1

#     # Now compute sum of intra-cluster distances
#     # sum_intra = sum_{k} sum_{i != j in cluster k} || x_i - x_j ||^2
#     #           = sum_{k} 2 * [ n_k * sum_sq(k) - || sum(k) ||^2 ]
#     # where n_k is cluster k size, sum_sq(k) is sum of norms^2 in cluster k,
#     # and sum(k) is the sum of vectors in cluster k.
#     sum_intra = torch.tensor(0.0, device=device)
#     n_intra_pairs = 0  # total # of i!=j pairs within clusters

#     for c in cluster_sums:
#         n_k = cluster_counts[c]
#         if n_k > 1:
#             s_k = cluster_sums[c]
#             sq_k = cluster_sqnorm_sums[c]
#             sum_intra_k = 2.0 * (n_k * sq_k - (s_k * s_k).sum())
#             sum_intra += sum_intra_k
#             n_intra_pairs += n_k * (n_k - 1)  # i != j pairs

#     # sum_inter = sum_total - sum_intra
#     # The total # of i != j pairs in the entire set is N * (N - 1)
#     # So the # of inter-cluster pairs is N*(N-1) - n_intra_pairs
#     sum_inter = sum_total - sum_intra
#     n_inter_pairs = N * (N - 1) - n_intra_pairs

#     # Mean distances
#     if n_intra_pairs > 0:
#         mean_intra = sum_intra / n_intra_pairs
#     else:
#         mean_intra = torch.tensor(0.0, device=device)

#     if n_inter_pairs > 0:
#         mean_inter = sum_inter / n_inter_pairs
#     else:
#         mean_inter = torch.tensor(0.0, device=device)

#     # Loss = mean_intra - alpha * mean_inter
#     loss = mean_intra - alpha * mean_inter
#     return loss

# import matplotlib.pyplot as plt
# import torch.nn.functional as F

# def compute_sim_dissim_loss(
#     embeddings: torch.Tensor,
#     user_ids_batch: list,
#     seed_users: list,
#     U_b_similar: dict,
#     U_b_dissimilar: dict,
#     tau: float = 0.07,
#     verbose: bool = False
# ) -> torch.Tensor:
#     device = embeddings.device
#     # L2 normalize embeddings for cosine similarity
#     embeddings = F.normalize(embeddings, dim=-1)
#     # Map user ID to embedding index
#     user2idx = {uid: i for i, uid in enumerate(user_ids_batch)}

#     total_loss = torch.tensor(0.0, device=device)
#     count = 0

#     sim_pos_list = []
#     sim_neg_list = []

#     for u in seed_users:
#         if u not in user2idx:
#             continue
#         z_u = embeddings[user2idx[u]].unsqueeze(0)          # [1, d]
#         pos_list = U_b_similar.get(u, [])
#         neg_idxs = [user2idx[n] for n in U_b_dissimilar.get(u, []) if n in user2idx]
#         if not neg_idxs:
#             continue

#         Z_neg = embeddings[neg_idxs]                       # [N_neg, d]
#         sim_neg = F.cosine_similarity(z_u, Z_neg, dim=-1)  # [N_neg]
#         sim_neg_list += sim_neg.detach().cpu().tolist()

#         for v in pos_list:
#             if v not in user2idx:
#                 continue
#             z_pos = embeddings[user2idx[v]].unsqueeze(0)    # [1, d]
#             sim_pos = F.cosine_similarity(z_u, z_pos, dim=-1)[0]
#             sim_pos_list.append(sim_pos.detach().cpu().item())

#             # Compute InfoNCE loss
#             sim_pos_scaled = sim_pos / tau
#             sim_neg_scaled = sim_neg / tau
#             num   = torch.exp(sim_pos_scaled)
#             denom = num + torch.exp(sim_neg_scaled).sum()
#             loss_uv = -torch.log(num / (denom + 1e-8))

#             total_loss += loss_uv
#             count += 1

#             if verbose:
#                 print(f"[u={u}, v={v}] sim_pos={sim_pos.item():.4f}, "
#                       f"sim_neg_mean={sim_neg.mean().item():.4f}, "
#                       f"loss_uv={loss_uv.item():.4f}, "
#                       f"neg_count={len(sim_neg)}")

#     if count == 0:
#         return torch.tensor(0.0, device=device)

#     avg_loss = total_loss / count
#     if verbose:
#         print(f"Total sim_dissim pairs: {count}, Avg Loss: {avg_loss.item():.4f}")
#         # Plot histogram of similarities
#         plt.figure(figsize=(8,4))
#         plt.hist(sim_pos_list, bins=50, alpha=0.6, label='pos')
#         plt.hist(sim_neg_list, bins=50, alpha=0.6, label='neg')
#         plt.legend()
#         plt.title("Positive vs Negative Cosine Similarities")
#         save_path = 'sim_dissim_hist-new.png'
#         plt.savefig(save_path)
#         print(f"[INFO] Saved similarity histogram to '{save_path}'")
#         plt.close()

#     return avg_loss



import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def compute_sim_dissim_loss(
    embeddings: torch.Tensor,
    user_ids_batch: list,
    seed_users: list,
    U_b_similar: dict,
    U_b_dissimilar: dict,
    tau: float = 0.07,
    verbose: bool = False
) -> torch.Tensor:
    """
    Compute a contrastive loss where each user's representation is mean‐pooled before
    measuring cosine similarity with positive and negative examples.

    Args:
        embeddings (torch.Tensor): Tensor of shape [batch_size, seq_len, d] or [batch_size, d].
                                   If embeddings.ndim == 3, we mean‐pool over dim=1.
        user_ids_batch (list): List of user IDs corresponding to the first dimension of embeddings.
        seed_users (list): Subset of user IDs to compute loss over.
        U_b_similar (dict): Mapping from user ID to a list of “similar” user IDs.
        U_b_dissimilar (dict): Mapping from user ID to a list of “dissimilar” user IDs.
        tau (float): Temperature parameter for InfoNCE.
        verbose (bool): Whether to print per-pair diagnostics and save a histogram.

    Returns:
        torch.Tensor: Scalar tensor representing the average contrastive loss.
    """
    device = embeddings.device

    # L2 normalize token‐level embeddings along the last dimension
    embeddings = F.normalize(embeddings, dim=-1)

    # If embeddings is [batch_size, seq_len, d], mean‐pool over seq_len → [batch_size, d]
    if embeddings.dim() == 3:
        pooled = embeddings.mean(dim=1)  # [batch_size, d]
    else:
        pooled = embeddings  # already [batch_size, d]

    # Build a quick lookup from user ID to row index
    user2idx = {uid: i for i, uid in enumerate(user_ids_batch)}

    total_loss = torch.tensor(0.0, device=device)
    count = 0

    sim_pos_list = []
    sim_neg_list = []

    for u in seed_users:
        if u not in user2idx:
            continue

        # Mean‐pooled embedding for user u: shape [1, d]
        z_u = pooled[user2idx[u]].unsqueeze(0)  # [1, d]

        # Get all candidate negatives (only keep those in the current mini‐batch)
        neg_idxs = [
            user2idx[n] for n in U_b_dissimilar.get(u, []) 
            if n in user2idx
        ]
        if not neg_idxs:
            continue

        # Stack and mean‐pooled negative embeddings: [N_neg, d]
        Z_neg = pooled[neg_idxs]  # [N_neg, d]
        sim_neg = F.cosine_similarity(z_u, Z_neg, dim=-1)  # [N_neg]
        sim_neg_list += sim_neg.detach().cpu().tolist()

        # For each positive v, compute loss against all negatives
        pos_list = U_b_similar.get(u, [])
        for v in pos_list:
            if v not in user2idx:
                continue

            # Mean‐pooled embedding for user v: [1, d]
            z_pos = pooled[user2idx[v]].unsqueeze(0)  # [1, d]
            sim_pos = F.cosine_similarity(z_u, z_pos, dim=-1)[0]
            sim_pos_list.append(sim_pos.detach().cpu().item())

            # Compute InfoNCE numerator and denominator
            sim_pos_scaled = sim_pos / tau                # scalar
            sim_neg_scaled = sim_neg / tau                # [N_neg]
            numerator = torch.exp(sim_pos_scaled)          # scalar
            denominator = numerator + torch.exp(sim_neg_scaled).sum()  # scalar
            loss_uv = -torch.log(numerator / (denominator + 1e-8))

            total_loss += loss_uv
            count += 1

            if verbose:
                neg_mean = sim_neg.mean().item() if sim_neg.numel() > 0 else 0.0
                print(
                    f"[u={u}, v={v}] sim_pos={sim_pos.item():.4f}, "
                    f"sim_neg_mean={neg_mean:.4f}, "
                    f"loss_uv={loss_uv.item():.4f}, neg_count={len(sim_neg)}"
                )

    if count == 0:
        return torch.tensor(0.0, device=device)

    avg_loss = total_loss / count

    if verbose:
        print(f"Total sim/dissim pairs: {count}, Avg Loss: {avg_loss.item():.4f}")

        # Plot histogram of positive vs. negative cosine similarities
        plt.figure(figsize=(8, 4))
        plt.hist(sim_pos_list, bins=50, alpha=0.6, label="pos")
        plt.hist(sim_neg_list, bins=50, alpha=0.6, label="neg")
        plt.legend()
        plt.title("Positive vs Negative Cosine Similarities")
        save_path = "sim_dissim_hist-new.png"
        plt.savefig(save_path)
        print(f"[INFO] Saved similarity histogram to '{save_path}'")
        plt.close()

    return avg_loss




def compute_cluster_loss_optimized(embeddings, cluster_labels, margin: float = 1.5, beta: float = 1.0):
    """
    Margin‐based cluster loss:
       L = L_intra + beta * L_rep

    L_intra = 1/N * sum_u ||z_u - c_{ℓ_u}||^2
    L_rep   = 2/(K*(K-1)) * sum_{p<q} max(0, margin - ||c_p - c_q||)^2

    embeddings:      [N, D] tensor
    cluster_labels:  [N]  long tensor, values in {0,...,K-1}
    margin:          排斥 margin m
    beta:            排斥项权重
    """
    device = embeddings.device
    N, D = embeddings.shape
    if N == 0:
        return torch.tensor(0.0, device=device)

    # 确保 labels 是 tensor 且在正确设备
    if not torch.is_tensor(cluster_labels):
        cluster_labels = torch.tensor(cluster_labels, dtype=torch.long, device=device)
    else:
        cluster_labels = cluster_labels.to(device=device, dtype=torch.long)

    # 1) 计算每个簇的中心
    unique_labels = torch.unique(cluster_labels)
    centers = []
    for c in unique_labels:
        mask = (cluster_labels == c)
        centers.append(embeddings[mask].mean(dim=0))
    centers = torch.stack(centers, dim=0)  # [K, D]
    K = centers.size(0)
    if K == 0:
        return torch.tensor(0.0, device=device)

    # 2) 簇内损失：节点到自己簇心的 MSE 平均
    #    构造从节点到其中心的索引映射
    label_to_idx = {int(c): i for i, c in enumerate(unique_labels)}
    idx_map = torch.tensor([label_to_idx[int(c)] for c in cluster_labels], device=device)
    node_centers = centers[idx_map]            # [N, D]
    L_intra = ((embeddings - node_centers) ** 2).sum(dim=1).mean()

    # 3) 簇心排斥项：对所有簇心对，margin‐based repulsion
    if K > 1:
        # 计算两两距离
        pdist = torch.cdist(centers, centers, p=2)    # [K, K]
        i, j = torch.triu_indices(K, K, offset=1)     # 上三角 (p<q)
        dists = pdist[i, j]                           # [K*(K-1)/2]
        # margin‐based: max(0, margin - dist)^2
        repulsion = torch.relu(margin - dists).pow(2).mean().mul(2.0)
    else:
        repulsion = torch.tensor(0.0, device=device)

    return L_intra + beta * repulsion  