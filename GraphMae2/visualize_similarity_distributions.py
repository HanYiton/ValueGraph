import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sampling import sample_and_expand_users

def load_user_dicts(user_map_path, similar_path, dissimilar_path):
    with open(user_map_path, "r") as f:
        user_map = json.load(f)
    with open(similar_path, "r") as f:
        similar_users_dict = json.load(f)
    with open(dissimilar_path, "r") as f:
        dissimilar_users_dict = json.load(f)
    return user_map, similar_users_dict, dissimilar_users_dict

def visualize_similarity_distributions(
    embeddings: torch.Tensor,
    user_ids_batch: list,
    seed_users: list,
    U_b_similar: dict,
    U_b_dissimilar: dict
):
    # Step 1: 归一化 embedding
    embeddings = F.normalize(embeddings, dim=-1)
    user2idx = {uid: i for i, uid in enumerate(user_ids_batch)}

    sim_pos_list = []
    sim_neg_list = []

    for u in seed_users:
        if u not in user2idx:
            continue
        z_u = embeddings[user2idx[u]].unsqueeze(0)  # [1, d]

        # 正样本
        for v in U_b_similar.get(u, []):
            if v not in user2idx:
                continue
            z_v = embeddings[user2idx[v]].unsqueeze(0)
            sim_uv = F.cosine_similarity(z_u, z_v, dim=-1).item()
            sim_pos_list.append(sim_uv)

        # 负样本
        for w in U_b_dissimilar.get(u, []):
            if w not in user2idx:
                continue
            z_w = embeddings[user2idx[w]].unsqueeze(0)
            sim_uw = F.cosine_similarity(z_u, z_w, dim=-1).item()
            sim_neg_list.append(sim_uw)

    # Step 2: 可视化
    plt.figure(figsize=(8, 5))
    sns.kdeplot(sim_pos_list, label='Positive (sim(u, v))', fill=True)
    sns.kdeplot(sim_neg_list, label='Negative (sim(u, w))', fill=True)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.title("Distribution of Positive vs Negative Similarity")
    plt.legend()
    plt.grid(True)
    plt.show()

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
    U_b, U_b_expanded, U_b_similar, U_b_dissimilar = sample_and_expand_users(
            user_map_path, similar_path, dissimilar_path,
            seed_batch_size=seed_batch_size, k=k, random_seed=42+epoch,
            user_map=user_map, similar_users_dict=similar_users_dict,
            dissimilar_users_dict=dissimilar_users_dict
        )
    visualize_similarity_distributions(
    embeddings=embeddings,
    user_ids_batch=user_ids_batch,
    seed_users=seed_users,
    U_b_similar=U_b_similar,
    U_b_dissimilar=U_b_dissimilar)