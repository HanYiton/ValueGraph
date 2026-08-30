import logging
import warnings
import numpy as np
import torch
from models.finetune import linear_probing_minibatch, finetune
from models import build_model
from datasets.lc_sampler import (
    setup_training_data,
)

from utils import (
    build_args,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


def evaluate(
    model,
    graph,
    feats,
    labels,
    split_idx,
    lr_f,
    weight_decay_f,
    max_epoch_f,
    linear_prob=True,
    device=0,
    batch_size=256,
    logger=None,
    ego_graph_nodes=None,
    label_rate=1.0,
    full_graph_forward=False,
    shuffle=True,
):
    logging.info("Using `lc` for evaluation...")
    num_train, num_val, num_test = [
        split_idx[k].shape[0] for k in ["train", "valid", "test"]
    ]
    print(num_train, num_val, num_test)

    train_g_idx = np.arange(0, num_train)
    val_g_idx = np.arange(num_train, num_train + num_val)
    test_g_idx = np.arange(num_train + num_val, num_train + num_val + num_test)

    train_ego_graph_nodes = [ego_graph_nodes[0][i] for i in range(len(train_g_idx))]
    val_ego_graph_nodes = [ego_graph_nodes[1][i] for i in range(len(val_g_idx))]
    test_ego_graph_nodes = [ego_graph_nodes[2][i] for i in range(len(test_g_idx))]
    train_lbls, val_lbls, test_lbls = (
        labels[train_g_idx],
        labels[val_g_idx],
        labels[test_g_idx],
    )

    # labels = [train_lbls, val_lbls, test_lbls]
    assert len(train_ego_graph_nodes) == len(train_lbls)
    assert len(val_ego_graph_nodes) == len(val_lbls)
    assert len(test_ego_graph_nodes) == len(test_lbls)

    print(f"num_train: {num_train}, num_val: {num_val}, num_test: {num_test}")
    logging.info(
        f"-- train_ego_nodes:{len(train_ego_graph_nodes)}, val_ego_nodes:{len(val_ego_graph_nodes)}, test_ego_nodes:{len(test_ego_graph_nodes)} ---"
    )

    if linear_prob:
        result = linear_probing_minibatch(
            model,
            graph,
            feats,
            [train_ego_graph_nodes, val_ego_graph_nodes, test_ego_graph_nodes],
            [train_lbls, val_lbls, test_lbls],
            lr_f=lr_f,
            weight_decay_f=weight_decay_f,
            max_epoch_f=max_epoch_f,
            batch_size=batch_size,
            device=device,
            shuffle=shuffle,
        )
    else:
        max_epoch_f = max_epoch_f // 2

        if label_rate < 1.0:
            rand_idx = np.arange(len(train_ego_graph_nodes))
            np.random.shuffle(rand_idx)
            rand_idx = rand_idx[: int(label_rate * len(train_ego_graph_nodes))]
            train_ego_graph_nodes = [train_ego_graph_nodes[i] for i in rand_idx]
            train_lbls = train_lbls[rand_idx]

        logging.info(
            f"-- train_ego_nodes:{len(train_ego_graph_nodes)}, val_ego_nodes:{len(val_ego_graph_nodes)}, test_ego_nodes:{len(test_ego_graph_nodes)} ---"
        )

        # train_lbls = (all_train_lbls, train_lbls)
        result = finetune(
            model,
            graph,
            feats,
            [train_ego_graph_nodes, val_ego_graph_nodes, test_ego_graph_nodes],
            [train_lbls, val_lbls, test_lbls],
            split_idx=split_idx,
            lr_f=lr_f,
            weight_decay_f=weight_decay_f,
            max_epoch_f=max_epoch_f,
            use_scheduler=True,
            batch_size=batch_size,
            device=device,
            logger=logger,
            full_graph_forward=full_graph_forward,
        )
    return result


if __name__ == "__main__":
    args = build_args()

    if args.device < 0:
        device = "cpu"
    else:
        device = "cuda:{}".format(args.device)
    full_graph_forward = (
        hasattr(args, "full_graph_forward")
        and args.full_graph_forward
        and not args.linear_prob
    )

    logging.info("---- start finetuning / evaluation ----")
    final_accs = []

    # loading data
    feats, graph, labels, split_idx, ego_graph_nodes = setup_training_data(
        args.dataset, args.data_dir
    )
    print(f"features size : {feats.shape[1]}")
    args.num_features = feats.shape[1]

    # loading model
    eval_model = build_model(args)
    eval_model.load_state_dict(torch.load(args.pretrain_path))
    eval_model.to(device)

    logging.info("start evaluation...")

    final_acc = evaluate(
        eval_model,
        graph,
        feats,
        labels,
        split_idx,
        args.lr,
        args.weight_decay,
        args.max_epoch_f,
        device=device,
        batch_size=args.batch_size_f,
        ego_graph_nodes=ego_graph_nodes,
        linear_prob=args.linear_prob,
        label_rate=args.label_rate,
        full_graph_forward=full_graph_forward,
        shuffle=True,
    )

    final_accs.append(float(final_acc))
    print(f"# final_acc: {np.mean(final_accs):.4f}, std: {np.std(final_accs):.4f}")
