import numpy as np
import torch
from tensorflow.keras.utils import to_categorical
from LSTM_models import LSTM_model_stance
from branch2treelabels import branch2treelabels_test
from utils_branch_reconstruction import reconstruct_branches

def build_user_features(user_ids, post_embeddings):
    """میانگین امبدینگ هر کاربر"""
    user_vectors = {}
    for i, uid in enumerate(user_ids):
        if uid not in user_vectors:
            user_vectors[uid] = []
        user_vectors[uid].append(post_embeddings[i])
    for uid in user_vectors:
        user_vectors[uid] = np.mean(user_vectors[uid], axis=0)
    return np.array([user_vectors[uid] for uid in user_ids])

def evaluation_function_stance_branchLSTM_RumEv(params):
    # --- 1) Load embeddings ---
    post_embeddings = torch.load("/workspace/stance_detect/RumourEval2019/post_embeddings_new.pt")
    post_embeddings = post_embeddings.detach().cpu().numpy()

    # --- 2) Load dataset ---
    y_train = np.load("preprocessing/saved_dataRumEval2019/train/fold_stance_labels.npy")
    y_test = np.load("preprocessing/saved_dataRumEval2019/test/fold_stance_labels.npy")
    ids_train = np.load("preprocessing/saved_dataRumEval2019/train/tweet_ids.npy")
    ids_test = np.load("preprocessing/saved_dataRumEval2019/test/tweet_ids.npy")
    user_ids_train = np.load("preprocessing/saved_dataRumEval2019/train/user_ids.npy")
    user_ids_test = np.load("preprocessing/saved_dataRumEval2019/test/user_ids.npy")

    # --- 3) Replace with your embeddings (align indices) ---
    x_train, branch_ids_train = reconstruct_branches(ids_train, post_embeddings[:len(ids_train)])
    x_test, branch_ids_test = reconstruct_branches(ids_test, post_embeddings[len(ids_train):len(ids_train)+len(ids_test)])

    # --- 4) User features ---
    user_features_train = build_user_features(user_ids_train, post_embeddings[:len(user_ids_train)])
    user_features_test = build_user_features(user_ids_test, post_embeddings[len(user_ids_train):len(user_ids_train)+len(user_ids_test)])

    # --- 5) Labels ---
    y_train_cat = to_categorical(y_train, num_classes=4)

    # --- 6) Train & Predict ---
    y_pred, confidence = LSTM_model_stance(
        x_train, y_train_cat, x_test, user_features_train, user_features_test, params, eval=True
    )

    # --- 7) Tree-level results ---
    trees, tree_prediction, tree_confidence = branch2treelabels_test(branch_ids_test, y_pred, confidence)
    return trees, tree_prediction, tree_confidence, y_test
