import os
import json
import numpy as np
import pandas as pd
import torch
import dgl
from ogb.nodeproppred import DglNodePropPredDataset

from .data_proc import preprocess, scale_feats
from .localclustering import step1_local_clustering
from utils import mask_edge

import logging
import torch.multiprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy("file_system")


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

# def collect_topk_ppr(graph, nodes, topk, alpha, epsilon):
#     if torch.is_tensor(nodes):
#         nodes = nodes.numpy()
#     row, col = graph.edges()
#     row = row.numpy()
#     col = col.numpy()
#     num_nodes = graph.num_nodes()

#     neighbors = build_topk_ppr((row, col), alpha, epsilon, nodes, topk, num_nodes=num_nodes)
#     return neighbors

# ---------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn.functional as F

def split_nodes(node_ids, train_ratio=0.8, val_ratio=0.1):
    node_ids = torch.tensor(node_ids) 
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


def load_dataset(data_dir, dataset_name, create_ego_graphs):

    id_mapping = {}

    # if dataset_name.startswith("social"):

    #     # TODO: add dataset processing step
    #     df_list = []
    #     torch_list = []

    #     for data in [
    #         'twibot'
    #         ]:
    #         file_path = os.path.join(data_dir, data)
    #         feat_path = f"{file_path}/twibot_modernbert_na.npy"
    #         text = pd.read_parquet(f"{file_path}/source_twitbot_no_na.parquet")
    #         # text = text.head(min(max_len, len(text)))
    #         feats = torch.tensor(np.load(feat_path))
    #         df_list.append(text)
    #         torch_list.append(feats)
        
    #     text = pd.concat(df_list, ignore_index=True)
    #     feats = torch.cat(torch_list, dim=0)
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     feats = feats.to(device)

    #     id_mapping = {old_id: new_id for new_id, old_id in enumerate(text['id'].unique())}
    #     text['id'] = text['id'].map(id_mapping)
    #     text['parent_id'] = text['parent_id'].map(id_mapping)

    #     print(f'Size: {len(text)}')
    #     text["label"] = np.random.choice([0, 1, 3], size=len(text))

    #     # train_idx, val_idx, test_idx = split_nodes(text["id"].astype(int).tolist())
    #     # split_idx = {"train": train_idx, "valid": val_idx, "test": test_idx}

    #     if 'split' in text.columns:
    #         text['split'] = text['split'].replace("dev", "test")
    #         split_idx = {
    #             split: torch.tensor(
    #                 text[text['split'] == split]['id'].tolist(), dtype=torch.int64
    #             ) if not text[text['split'] == split].empty else torch.empty(0, dtype=torch.int64)
    #             for split in ["train", "test", "valid"]
    #         }

    #         # If "valid" is empty, use "test" as "valid"
    #         if split_idx["valid"].numel() == 0:
    #             split_idx["valid"] = split_idx["test"]

    #     # if split does not exists then apply the same dataset for inference
    #     else:
    #         split_idx = {
    #             split: torch.tensor(
    #                 text['id'].tolist(), dtype=torch.int64
    #             )
    #             for split in ["train", "test", "valid"]
    #         }
    #         # just arbitary
    #         split_idx["test"] = split_idx["test"][:100, ]
    #         split_idx["valid"] = split_idx["valid"][:100,]

    #     label = torch.tensor(text["label"].tolist()).to(torch.long)
    #     edges = text[['id', 'parent_id']].copy()
    #     edges = edges[edges['parent_id'].notnull()]
    #     edges.columns = ['source', 'target']
    #     source = torch.tensor([int(x) for x in edges["source"].tolist()])
    #     target = torch.tensor([int(x) for x in edges["target"].tolist()])
    
    #     # graph construction
    #     graph = dgl.graph((source, target))
    #     num_nodes = graph.num_nodes()
    #     num_feats = feats.size(0)
    #     if num_feats > num_nodes:
    #         n_extra = num_feats - num_nodes
    #         print(f"[INFO] Graph has {num_nodes} nodes but {num_feats} feature rows; adding {n_extra} isolated nodes.")
    #         graph.add_nodes(n_extra)
    #     elif num_feats < num_nodes:
    #         raise ValueError(f"Fewer feature rows ({num_feats}) than graph nodes ({num_nodes}).")

    #     # Preprocess graph
    #     graph = preprocess(graph).to(device)
    #     graph = graph.to(feats.device)
    if dataset_name.startswith("social"):

        # Load text and features
        df_list = []
        torch_list = []

        for data in ['enron_spam_data']:
            file_path = os.path.join(data_dir, data)
            feat_path = f"{file_path}/Enron_content.npy"
            text = pd.read_csv(f"{file_path}/merged_inner.csv")
            feats = torch.tensor(np.load(feat_path))
            df_list.append(text)
            torch_list.append(feats)
        
        # Concatenate data
        text = pd.concat(df_list, ignore_index=True)
        feats = torch.cat(torch_list, dim=0)

        # Determine device and move features
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feats = feats.to(device)

        # Remap IDs
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(text['message_id'].unique())}
        text['message_id'] = text['message_id'].map(id_mapping)
        text['child_message_id'] = text['child_message_id'].map(id_mapping)

        print(f'Size: {len(text)}')
        text["label"] = np.random.choice([0, 1, 3], size=len(text))

        # Prepare splits
        if 'split' in text.columns:
            text['split'] = text['split'].replace("dev", "test")
            split_idx = {
                split: torch.tensor(
                    text[text['split'] == split]['message_id'].tolist(), dtype=torch.int64
                ) if not text[text['split'] == split].empty else torch.empty(0, dtype=torch.int64)
                for split in ["train", "test", "valid"]
            }
            if split_idx["valid"].numel() == 0:
                split_idx["valid"] = split_idx["test"]
        else:
            all_ids = torch.tensor(text['message_id'].tolist(), dtype=torch.int64).to(device)
            split_idx = {split: all_ids for split in ["train", "test", "valid"]}
            split_idx["test"] = split_idx["test"][:100]
            split_idx["valid"] = split_idx["valid"][:100]

        # Labels
        label = torch.tensor(text["label"].tolist(), dtype=torch.long).to(device)

        # Build edges
        edges = text[['message_id', 'child_message_id']].dropna().rename(columns={'message_id':'source','child_message_id':'target'})
        source = torch.tensor(edges['source'].astype(int).tolist(), dtype=torch.int64)
        target = torch.tensor(edges['target'].astype(int).tolist(), dtype=torch.int64)

        # Graph construction and move to device
        graph = dgl.graph((source, target))
        num_nodes = graph.num_nodes()
        num_feats = feats.size(0)
        if num_feats > num_nodes:
            n_extra = num_feats - num_nodes
            print(f"[INFO] Graph has {num_nodes} nodes but {num_feats} feature rows; adding {n_extra} isolated nodes.")
            graph.add_nodes(n_extra)
        elif num_feats < num_nodes:
            raise ValueError(f"Fewer feature rows ({num_feats}) than graph nodes ({num_nodes}).")

        graph = preprocess(graph).to(device)

    
    else:
        file_path = os.path.join(data_dir, dataset_name)
        feat_path = f"{file_path}/feat_3165.npy"
        text = pd.read_parquet(f"{file_path}/source.parquet")

        # feat = np.vstack(
        #     [
        #         np.load("/workspace/data/finetune/twibot/embeddings/twibot_feat.npy"),
        #         np.load("/workspace/data/finetune/twibot/embeddings/twibot_feat_1.npy"),
        #     ]
        # )
        # print(len(feat), len(text))
        # np.save("/workspace/data/finetune/twibot/embeddings/feat.npy", feat)

        # text['author_id'] = text['author_id'].apply(lambda x: 'u' + str(x))
        # user_map = text.groupby("author_id")['id'].apply(list).to_dict()
        # with open(os.path.join(file_path, 'embeddings', 'user_map.json'), "w") as f:
        #     json.dump(user_map, f)

        # import ipdb
        # ipdb.set_trace()

        # 21315584, 16392192, 16384000, 
        # 8192000, 8192000, 8192000, 8192000, 3_00000
        start = 0
        end = 5_000_000
        feats = torch.tensor(np.load(feat_path)[start: start + end])

        start = 54091776 + 34125681
        text = text.iloc[start: start + len(feats)]

        print(f'Data Size: {len(text)}')

        # read label
        label_path = f"{file_path}/label.parquet"
        if os.path.exists(label_path):
            label = pd.read_parquet(label_path)
            text = text.merge(
                label[['node_id', 'class_id', 'split']], 
                left_on='id', 
                right_on='node_id', 
                how='left'
            )
            text['class_id'] = text['class_id'].fillna(-1)
        else:
            # random label
            text["class_id"] = np.random.choice([0, 1], size=len(text))

        label = torch.tensor(text["class_id"].tolist()).to(torch.long)
        # map id in order to start from 0
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(text['id'].unique())}
        text['id'] = text['id'].map(id_mapping)
        text['parent_id'] = text['parent_id'].map(id_mapping)

        if 'split' in text.columns:
            text['split'] = text['split'].replace("dev", "test")
            split_idx = {
                split: torch.tensor(
                    text[text['split'] == split]['id'].tolist(), dtype=torch.int64
                ) if not text[text['split'] == split].empty else torch.empty(0, dtype=torch.int64)
                for split in ["train", "test", "valid"]
            }

            # If "valid" is empty, use "test" as "valid"
            if split_idx["valid"].numel() == 0:
                split_idx["valid"] = split_idx["test"]

        # if split does not exists then apply the same dataset for inference
        else:
            split_idx = {
                split: torch.tensor(
                    text['id'].tolist(), dtype=torch.int64
                )
                for split in ["train", "test", "valid"]
            }
            # just arbitary
            split_idx["test"] = split_idx["test"][:100, ]
            split_idx["valid"] = split_idx["valid"][:100, ]

            # all_ids = torch.tensor(text['id'].tolist(), dtype=torch.int64)
            # split_idx = {split: all_ids for split in ["train", "test", "valid"]}
        
        # form edges
        # First, set parent_id = id where both conditions are False
        # set single node parent's id to itself
        _ids = set(text["id"])
        _parent_ids = set(text["parent_id"])
        print(len(text[text["parent_id"].notnull()]))
        print(len([x for x in _parent_ids if x in _ids]))
        mask = ~text["id"].isin(_parent_ids) & ~text["parent_id"].isin(_ids)
        text.loc[mask, "parent_id"] = text.loc[mask, "id"]

        edges = text[['id', 'parent_id']].copy()
        edges = edges[edges['parent_id'].notnull()]
        edges.columns = ['source', 'target']
        source = torch.tensor([int(x) for x in edges["source"].tolist()])
        target = torch.tensor([int(x) for x in edges["target"].tolist()])
        
        # graph construction
        graph = dgl.graph((source, target))
        num_nodes = graph.num_nodes()
        num_feats = feats.size(0)
        if num_feats > num_nodes:
            n_extra = num_feats - num_nodes
            print(f"[INFO] Graph has {num_nodes} nodes but {num_feats} feature rows; adding {n_extra} isolated nodes.")
            graph.add_nodes(n_extra)
        elif num_feats < num_nodes:
            raise ValueError(f"Fewer feature rows ({num_feats}) than graph nodes ({num_nodes}).")

        # Preprocess graph
        graph = preprocess(graph).to(device)
        graph = graph.to(feats.device)
        num_nodes = graph.num_nodes()

        # if split_idx:
        #     train_mask = torch.full((num_nodes,), False).index_fill_(0, split_idx['train'], True)
        #     val_mask = torch.full((num_nodes,), False).index_fill_(0, split_idx['valid'], True)
        #     test_mask = torch.full((num_nodes,), False).index_fill_(0, split_idx['test'], True)
        #     graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = train_mask, val_mask, test_mask

    # df_user = text.copy()
    # cond = (df_user['userid'] == '[deleted]') | (df_user['userid'].str.startswith('---'))
    # df_user = df_user[~cond]
    # df_user['user_count'] = df_user.groupby('userid')['id'].transform('count').values
    # df_user = df_user.query("user_count >= 10 and user_count <= 1000")
    # user_map = df_user.groupby("userid")['id'].apply(list).to_dict()
    # file_path = data_dir if dataset_name == 'social' else file_path
    # with open(os.path.join(file_path, 'user_map.json'), "w") as f:
        # json.dump(user_map, f)
    # print(f'Saved {len(user_map)} user map to {file_path}')

    if create_ego_graphs:
        step1_local_clustering(
            graph,
            dataset_name,
            split_idx,
            16,
            10,
            100,
            8,
            "acl",
            f"lc_ego_graphs/{dataset_name}",
        )

    return feats, graph, label, split_idx, id_mapping


class LinearProbingDataLoader(DataLoader):
    def __init__(self, idx, feats, labels=None, **kwargs):
        self.labels = labels
        self.feats = feats

        kwargs["collate_fn"] = self.__collate_fn__
        super().__init__(dataset=idx, **kwargs)

    def __collate_fn__(self, batch_idx):
        feats = self.feats[batch_idx]
        label = self.labels[batch_idx]

        return feats, label


class OnlineLCLoader(DataLoader):
    def __init__(
        self, root_nodes, graph, feats, labels=None, drop_edge_rate=0, **kwargs
    ):
        self.graph = graph
        self.labels = labels
        self._drop_edge_rate = drop_edge_rate
        self.ego_graph_nodes = root_nodes
        self.feats = feats

        dataset = np.arange(len(root_nodes))
        kwargs["collate_fn"] = self.__collate_fn__
        super().__init__(dataset, **kwargs)

    def drop_edge(self, g):
        if self._drop_edge_rate <= 0:
            return g, g

        g = g.remove_self_loop()
        mask_index1 = mask_edge(g, self._drop_edge_rate)
        mask_index2 = mask_edge(g, self._drop_edge_rate)
        g1 = dgl.remove_edges(g, mask_index1).add_self_loop()
        g2 = dgl.remove_edges(g, mask_index2).add_self_loop()
        return g1, g2

    def __collate_fn__(self, batch_idx):
        ego_nodes = [self.ego_graph_nodes[i] for i in batch_idx]
        subgs = [self.graph.subgraph(ego_nodes[i]) for i in range(len(ego_nodes))]
        sg = dgl.batch(subgs)

        nodes = torch.from_numpy(np.concatenate(ego_nodes)).long()
        num_nodes = [x.shape[0] for x in ego_nodes]
        cum_num_nodes = np.cumsum([0] + num_nodes)[:-1]

        if self._drop_edge_rate > 0:
            drop_g1, drop_g2 = self.drop_edge(sg)

        sg = sg.remove_self_loop().add_self_loop()
        sg.ndata["feat"] = self.feats[nodes]
        targets = torch.from_numpy(cum_num_nodes)

        if self.labels != None:
            label = self.labels[batch_idx]
        else:
            label = None

        if self._drop_edge_rate > 0:
            return sg, targets, label, nodes, drop_g1, drop_g2
        else:
            return sg, targets, label, nodes


def setup_training_data(dataset_name, data_dir):

    ego_graphs_file_path = f"lc_ego_graphs/{dataset_name}/graphs.pt"
    create_ego_graphs = False if os.path.exists(ego_graphs_file_path) else True

    print(f'Create Cluster Graph: {create_ego_graphs}')
    feats, graph, labels, split_idx, id_mapping = load_dataset(
        data_dir, dataset_name, create_ego_graphs=create_ego_graphs
    )

    train_lbls = labels[split_idx["train"]]
    valid_lbls = labels[split_idx["valid"]]
    test_lbls = labels[split_idx["test"]]
    labels = torch.cat([train_lbls, valid_lbls, test_lbls])
    os.makedirs(os.path.dirname(ego_graphs_file_path), exist_ok=True)

    if not os.path.exists(ego_graphs_file_path):
        raise FileNotFoundError(f"{ego_graphs_file_path} doesn't exist")
    else:
        nodes = torch.load(ego_graphs_file_path, weights_only=False)
        # nodes = torch.load(ego_graphs_file_path)

    return feats, graph, labels, split_idx, nodes


def setup_training_dataloder(
    loader_type,
    training_nodes,
    graph,
    feats,
    batch_size,
    drop_edge_rate=0,
    pretrain_clustergcn=False,
    cluster_iter_data=None,
):
    num_workers = 8

    if loader_type == "lc":
        assert training_nodes is not None
    else:
        raise NotImplementedError(f"{loader_type} is not implemented yet")

    # print(" -------- drop edge rate: {} --------".format(drop_edge_rate))
    dataloader = OnlineLCLoader(
        training_nodes,
        graph,
        feats=feats,
        drop_edge_rate=drop_edge_rate,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        persistent_workers=True,
        num_workers=num_workers,
    )

    return dataloader


def setup_eval_dataloder(
    loader_type, graph, feats, ego_graph_nodes=None, batch_size=128, shuffle=False
):
    num_workers = 8
    if loader_type == "lc":
        assert ego_graph_nodes is not None
    else:
        raise NotImplementedError(f"{loader_type} is not implemented yet")

    dataloader = OnlineLCLoader(
        ego_graph_nodes,
        graph,
        feats,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        persistent_workers=True,
        num_workers=num_workers,
    )
    return dataloader


def setup_finetune_dataloder(
    loader_type, graph, feats, ego_graph_nodes, labels, batch_size, shuffle=False
):
    num_workers = 8

    if loader_type == "lc":
        assert ego_graph_nodes is not None
    else:
        raise NotImplementedError(f"{loader_type} is not implemented yet")

    dataloader = OnlineLCLoader(
        ego_graph_nodes,
        graph,
        feats,
        labels=labels,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    return dataloader
