import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset

from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F


def split_nodes(node_ids, train_ratio=0.8, val_ratio=0.1):
    node_ids = torch.tensor(node_ids)  # Ensure it's a tensor
    num_nodes = len(node_ids)

    # Shuffle the node IDs
    shuffled_indices = torch.randperm(num_nodes)

    # Compute split sizes
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    test_size = num_nodes - train_size - val_size  # Remaining for test

    # Split the indices
    train_idx = shuffled_indices[:train_size]
    val_idx = shuffled_indices[train_size : train_size + val_size]
    test_idx = shuffled_indices[train_size + val_size :]

    return train_idx, val_idx, test_idx


GRAPH_DICT = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset,
    "ogbn-arxiv": DglNodePropPredDataset,
}


def load_small_dataset(dataset_name):

    raise NotImplementedError

    # # assert dataset_name in GRAPH_DICT, f"Unknow dataset: {dataset_name}."
    # if dataset_name.startswith("ogbn"):
    #     dataset = GRAPH_DICT[dataset_name](dataset_name)
    # else:
    #     # dataset = GRAPH_DICT[dataset_name]()
    #     pass

    # if dataset_name == "ogbn-arxiv":
    #     # python main_full_batch.py --dataset ogbn-arxiv --encoder gat --decoder gat --seed 0 --device 0

    #     graph, labels = dataset[0]
    #     num_nodes = graph.num_nodes()

    #     split_idx = dataset.get_idx_split()
    #     train_idx, val_idx, test_idx = (
    #         split_idx["train"],
    #         split_idx["valid"],
    #         split_idx["test"],
    #     )
    #     graph = preprocess(graph)

    #     if not torch.is_tensor(train_idx):
    #         train_idx = torch.as_tensor(train_idx)
    #         val_idx = torch.as_tensor(val_idx)
    #         test_idx = torch.as_tensor(test_idx)

    #     feat = graph.ndata["feat"]
    #     feat = scale_feats(feat)
    #     graph.ndata["feat"] = feat

    #     train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
    #     val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
    #     test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
    #     graph.ndata["label"] = labels.view(-1)
    #     graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = (
    #         train_mask,
    #         val_mask,
    #         test_mask,
    #     )

    # elif dataset_name.startswith("social"):
    #     file_path = f"data/social/{dataset_name}"

    #     edge_index = pd.read_csv(f"{file_path}/edges.csv")
    #     text = pd.read_csv(f"{file_path}/text.csv")
    #     train_idx, val_idx, test_idx = split_nodes(text["node_id"].tolist())
    #     text["label"] = np.random.choice([0, 1], size=len(text))

    #     label = torch.tensor(text["label"].tolist()).to(torch.long)
    #     source = torch.tensor([int(x) for x in edge_index["source"].tolist()])
    #     target = torch.tensor([int(x) for x in edge_index["target"].tolist()])

    #     # graph construction
    #     graph = dgl.graph((source, target))
    #     # graph = dgl.remove_self_loop(graph)
    #     # graph = dgl.add_self_loop(graph)
    #     graph = preprocess(graph)

    #     # feature encoding
    #     batch_size = 2048
    #     sentences = text["text"].tolist()
    #     sentence_batches = np.array_split(
    #         sentences, np.ceil(len(sentences) / batch_size)
    #     )
    #     embeddings = [
    #         encode(batch) for batch in tqdm(sentence_batches, desc="Encoding batches")
    #     ]
    #     feats = np.vstack([emb.cpu().numpy() for emb in embeddings])
    #     feats = torch.tensor(feats)

    #     # feat = scale_feats(feat)
    #     graph.ndata["feat"] = feats
    #     graph.ndata["label"] = label

    #     # add graph mask
    #     num_nodes = graph.num_nodes()
    #     train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
    #     val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
    #     test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
    #     graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = (
    #         train_mask,
    #         val_mask,
    #         test_mask,
    #     )

    # else:
    #     graph = dataset[0]
    #     graph = graph.remove_self_loop()
    #     graph = graph.add_self_loop()

    # num_features = graph.ndata["feat"].shape[1]
    # # num_classes = dataset.num_classes
    # num_classes = 2

    # return graph, (num_features, num_classes)


def preprocess(graph):
    # make bidirected
    if "feat" in graph.ndata:
        feat = graph.ndata["feat"]
    else:
        feat = None
    # src, dst = graph.all_edges()
    # graph.add_edges(dst, src)
    graph = dgl.to_bidirected(graph)
    if feat is not None:
        graph.ndata["feat"] = feat

    # add self-loop
    graph = graph.remove_self_loop().add_self_loop()
    # graph.create_formats_()
    return graph


def scale_feats(x):
    logging.info("### scaling features ###")
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats
