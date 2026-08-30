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

import torch
import torch.nn.functional as F
from collections import defaultdict

def sample_and_expand_users(
    user_map_path=None,
    similar_path=None,
    dissimilar_path=None,
    seed_batch_size=10,
    k=5,
    random_seed=42,
    user_map=None,
    similar_users_dict=None,
    dissimilar_users_dict=None,
    min_similar=1,
    min_dissimilar=1,
    normalize_input=True,
    verbose=True
):
    """
    Optimized version to sample seed users and expand with similar/dissimilar users.
    Now avoids failures by validating sim/dissim overlap during filtering.
    """
    start_time = time.time()

    # --- 1) Load if not preloaded ---
    if user_map is None:
        with open(user_map_path, 'r') as f:
            user_map = json.load(f)
    if similar_users_dict is None:
        with open(similar_path, 'r') as f:
            similar_users_dict = json.load(f)
    if dissimilar_users_dict is None:
        with open(dissimilar_path, 'r') as f:
            dissimilar_users_dict = json.load(f)

    # --- 2) Normalize all user IDs to strings ---
    if normalize_input:
        user_map = {str(k): v for k, v in user_map.items()}
        similar_users_dict = {str(k): [str(n) for n in v] for k, v in similar_users_dict.items()}
        dissimilar_users_dict = {str(k): [str(n) for n in v] for k, v in dissimilar_users_dict.items()}

    valid_user_set = set(user_map.keys())
    random.seed(random_seed)

    # --- 3) Filter neighbors ---
    def filter_neighbors(neighbors_dict):
        return {
            user: [nbr for nbr in nbrs if nbr in valid_user_set and nbr != user]
            for user, nbrs in neighbors_dict.items()
            if user in valid_user_set
        }

    filtered_similar = filter_neighbors(similar_users_dict)
    filtered_dissimilar = filter_neighbors(dissimilar_users_dict)

    # --- 4) Check candidates AFTER dissimilar overlap removal ---
    candidate_users = set(filtered_similar.keys()) & set(filtered_dissimilar.keys())
    valid_users = []
    for user in candidate_users:
        sim = set(filtered_similar.get(user, []))
        dissim = set(filtered_dissimilar.get(user, [])) - sim
        if len(sim) >= min_similar and len(dissim) >= min_dissimilar:
            valid_users.append(user)

    if len(valid_users) < seed_batch_size:
        if verbose:
            print(
                f"Warning: Only {len(valid_users)} users have >= {min_similar} similar and >= {min_dissimilar} dissimilar after filtering overlap. "
                f"Adjusting seed_batch_size from {seed_batch_size} to {len(valid_users)}."
            )
        seed_batch_size = len(valid_users)

    if seed_batch_size == 0:
        raise ValueError("No users have enough similar/dissimilar neighbors. Check data.")

    # --- 5) Sample seeds & prepare data ---
    U_b = random.sample(valid_users, seed_batch_size)
    U_b_similar = {}
    U_b_dissimilar = {}
    expanded_users = set(U_b)

    for user in U_b:
        sim_candidates = filtered_similar[user]
        dissim_candidates = [d for d in filtered_dissimilar[user] if d not in sim_candidates]

        sim_selected = random.sample(sim_candidates, min(k, len(sim_candidates)))
        dissim_selected = random.sample(dissim_candidates, min(k, len(dissim_candidates)))

        # No need for another min_similar/min_dissimilar check here anymore
        U_b_similar[user] = sim_selected
        U_b_dissimilar[user] = dissim_selected
        expanded_users.update(sim_selected)
        expanded_users.update(dissim_selected)

        if verbose and (len(sim_selected) < k or len(dissim_selected) < k):
            print(f"Note: User '{user}' has only {len(sim_selected)} similar and {len(dissim_selected)} dissimilar (requested k={k}).")

    U_b_expanded = list(expanded_users)

    if verbose:
        print(
            f"Sampled {len(U_b)} seed users. Expanded set size: {len(U_b_expanded)}. "
            f"Took {time.time() - start_time:.2f} seconds."
        )

    return U_b, U_b_expanded, U_b_similar, U_b_dissimilar