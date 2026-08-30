import torch
# torch.cuda.set_device(0)
import torch.nn.functional as F
import torch.nn as nn
import random
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
import os, time, matplotlib.pyplot as plt

from infer_batch import id_mapping, feats, graph, labels, split_idx, model, data_loader
from losses import compute_user_level_contrast_loss, compute_sim_dissim_loss, compute_cluster_loss_optimized
from sampling import sample_and_expand_users


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
    lambda_cluster=1.0, lambda_sim=100, lambda_user=0.5, K=10, kmeans_update_freq=1, lr=1e-4
):
    loss_records = []
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    best_loss = float("inf")
    checkpoint_path = "workspace/baseline/GraphMae2/our_checkpoints/best_model3.pt"

    for epoch in tqdm(range(1, epochs + 1)):
        # 1) 采样用户及其正负集
        U_b, U_b_expanded, U_b_similar, U_b_dissimilar = sample_and_expand_users(
            user_map_path, similar_path, dissimilar_path,
            seed_batch_size=seed_batch_size, k=k, random_seed=42+epoch,
            user_map=user_map, similar_users_dict=similar_users_dict,
            dissimilar_users_dict=dissimilar_users_dict
        )

        # 2) 从采样用户映射到 post 节点 index
        expanded_node_indices = []
        for u in U_b_expanded:
            val = user_map.get(u, [])
            if isinstance(val, list): expanded_node_indices.extend(val)
            elif val: expanded_node_indices.append(val)
        for sim_list in U_b_similar.values():
            for u in sim_list:
                val = user_map.get(u, [])
                if isinstance(val, list): expanded_node_indices.extend(val)
                elif val: expanded_node_indices.append(val)
        for diss_list in U_b_dissimilar.values():
            for u in diss_list:
                val = user_map.get(u, [])
                if isinstance(val, list): expanded_node_indices.extend(val)
                elif val: expanded_node_indices.append(val)
        expanded_node_indices = list(set(expanded_node_indices))
        expanded_node_indices_int = [id_mapping[str(x)] for x in expanded_node_indices if str(x) in id_mapping]
        expanded_node_indices_int = expanded_node_indices_int[:20_000]

        # 3) 迭代数据 loader，收集 embeddings
        lis, node_indices = [], []
        for batch in data_loader:
            batch_g, targets, batch_lbls, node_idx = batch
            node_idx = node_idx.to(device)
            mask = torch.tensor([idx.item() in expanded_node_indices_int for idx in node_idx[targets]], device=device)
            if not mask.any(): continue
            batch_g = batch_g.to(device)
            x = batch_g.ndata.pop("feat")
            prediction = model(batch_g, x)
            batch_emb = prediction[targets][mask]
            lis.append(batch_emb)
            node_indices.extend(node_idx[targets][mask].tolist())
            del prediction, x, batch_g, batch_lbls, mask
            torch.cuda.empty_cache()

        expanded_embeddings = torch.cat(lis, dim=0)
        inverse_map = {v: k for k, v in id_mapping.items()}
        post_to_user = {p: u for u, posts in user_map.items() for p in posts}
        user_ids_batch, valid_embs = [], []
        for i, nid in enumerate(node_indices):
            post_id = inverse_map.get(nid)
            uid = post_to_user.get(post_id)
            if uid:
                user_ids_batch.append(uid)
                valid_embs.append(expanded_embeddings[i])
        expanded_embeddings = torch.stack(valid_embs)
        print(f"[Epoch {epoch}] Top user frequencies: {Counter(user_ids_batch).most_common(5)}", flush=True)

        # 4) 计算对比损失（首 epoch 打印详细信息）
        verbose_flag = (epoch == 1)
        L_sim_dissim = compute_sim_dissim_loss(
            expanded_embeddings, user_ids_batch, U_b, U_b_similar, U_b_dissimilar,
            tau=tau, verbose=verbose_flag
        )
        cluster_labels = run_kmeans(expanded_embeddings, num_clusters=K)
        L_cluster = compute_cluster_loss_optimized(
    expanded_embeddings,
    cluster_labels,
    margin=1.0,    # 你可以从 0.5~2.0 尝试
    beta=0.5)
        total_loss = L_sim_dissim +lambda_cluster*L_cluster# 目前仅调试这一项

        # DEBUG: 打印 loss
        print(f"[DEBUG] Epoch {epoch} before backward: total_loss={total_loss.item():.6f}", flush=True)

        optimizer.zero_grad()
        total_loss.backward()

        # DEBUG: 打印梯度范数及 zero-gradient 参数数
        total_norm = 0.0
        zero_count = 0
        for name, param in model.named_parameters():
            if param.grad is None:
                zero_count += 1
            else:
                norm = param.grad.data.norm(2)
                total_norm += norm.item() ** 2
        total_norm = total_norm ** 0.5
        print(f"[DEBUG] Step {global_step+1}: gradient norm = {total_norm:.6f}, zero_grad_params = {zero_count}", flush=True)

        optimizer.step()
        global_step += 1

        # 5) 记录并可视化 loss 曲线
        loss_records.append([epoch, L_sim_dissim.item(),L_cluster.item(), total_loss.item()])
        os.makedirs("plots_sim_cluster", exist_ok=True)
        arr = np.array(loss_records)
        plt.figure()
        plt.plot(arr[:, 0], arr[:, 1], label='L_sim_dissim')
        plt.plot(arr[:, 0],lambda_cluster*arr[:, 2], label='L_cluster')
        plt.plot(arr[:, 0], arr[:, 3], label='Total Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss Curve up to Epoch {epoch}")
        plt.grid(True)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(f"plots_sim_cluster/loss_epoch_{epoch}.png")
        plt.close()

        # 6) 保存最优模型
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'global_step': global_step
            }, checkpoint_path)
            print(f"[DEBUG] Saved new best model at step {global_step} with loss {best_loss:.6f}")

        # 清理
        del expanded_embeddings
        torch.cuda.empty_cache()


if __name__ == "__main__":
    user_map_path = 'workspace/data/pretrain/user_map_combined.json'
    similar_path = "workspace/data/pretrain/sim_global99_9.json"
    dissimilar_path = "workspace/data/pretrain/dissim_global0_1.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    user_map, sim_dict, dissim_dict = load_user_dicts(user_map_path, similar_path, dissimilar_path)
    train_contrastive_clustering1(
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
        epochs=60,
        seed_batch_size=3000,
        k=20,
        tau=0.07,
        lambda_cluster=0.05,
        lambda_sim=10,
        lambda_user=1,
        K=5,
        lr=1e-3)
