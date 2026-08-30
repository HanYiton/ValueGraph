import logging
import os
import numpy as np
from tqdm import tqdm

import torch

from utils import (
    WandbLogger,
    build_args,
    create_optimizer,
    set_random_seed,
    load_best_configs,
    show_occupied_memory,
)
from models import build_model
from datasets.lc_sampler import (
    setup_training_dataloder,
    setup_training_data,
)
from models.finetune import linear_probing_minibatch, finetune

import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


def pretrain(
    model,
    feats,
    graph,
    ego_graph_nodes,
    max_epoch,
    device,
    use_scheduler,
    lr,
    weight_decay,
    batch_size=512,
    sampling_method="lc",
    optimizer="adam",
    drop_edge_rate=0,
):
    logging.info("start training..")

    model = model.to(device)
    optimizer = create_optimizer(optimizer, model, lr, weight_decay)
    
    dataloader = setup_training_dataloder(
        sampling_method,
        ego_graph_nodes,
        graph,
        feats,
        batch_size=batch_size,
        drop_edge_rate=drop_edge_rate,
    )

    logging.info(f"After creating dataloader: Memory: {show_occupied_memory():.2f} MB")
    if use_scheduler and max_epoch > 0:
        logging.info("Use scheduler")
        scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None

    for epoch in range(max_epoch):
        epoch_iter = tqdm(dataloader)
        losses = []
        # assert (graph.in_degrees() > 0).all(), "after loading"

        for batch_g in epoch_iter:
            model.train()
            if drop_edge_rate > 0:
                batch_g, targets, _, node_idx, drop_g1, drop_g2 = batch_g
                batch_g = batch_g.to(device)
                drop_g1 = drop_g1.to(device)
                drop_g2 = drop_g2.to(device)
                x = batch_g.ndata.pop("feat")
                loss = model(batch_g, x, targets, epoch, drop_g1, drop_g2)
            else:
                batch_g, targets, _, node_idx = batch_g
                batch_g = batch_g.to(device)
                x = batch_g.ndata.pop("feat")
                loss = model(batch_g, x, targets, epoch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()

            epoch_iter.set_description(
                f"train_loss: {loss.item():.4f}, Memory: {show_occupied_memory():.2f} MB"
            )
            losses.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        torch.save(model.state_dict(), os.path.join(model_dir, model_name))

        print(
            f"# Epoch {epoch} | train_loss: {np.mean(losses):.4f}, Memory: {show_occupied_memory():.2f} MB"
        )

    return model


if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args)

    if args.device < 0:
        device = "cpu"
    else:
        device = "cuda:{}".format(args.device)
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    encoder = args.encoder
    decoder = args.decoder
    num_hidden = args.num_hidden
    drop_edge_rate = args.drop_edge_rate

    optim_type = args.optimizer
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    no_pretrain = args.no_pretrain
    logs = args.logging
    use_scheduler = args.scheduler
    batch_size = args.batch_size
    batch_size_f = args.batch_size_f
    sampling_method = args.sampling_method
    ego_graph_file_path = args.ego_graph_file_path
    data_dir = args.data_dir

    n_procs = 1
    optimizer_type = args.optimizer
    label_rate = args.label_rate
    lam = args.lam
    full_graph_forward = (
        hasattr(args, "full_graph_forward")
        and args.full_graph_forward
        and not linear_prob
    )

    model_dir = "checkpoints"
    os.makedirs(model_dir, exist_ok=True)

    set_random_seed(0)
    print(args)

    logging.info(
        f"Before loading data, occupied memory: {show_occupied_memory():.2f} MB"
    )  # in MB
    feats, graph, labels, split_idx, ego_graph_nodes = setup_training_data(
        dataset_name, data_dir
    )
    if dataset_name == "ogbn-papers100M":
        pretrain_ego_graph_nodes = (
            ego_graph_nodes[0]
            + ego_graph_nodes[1]
            + ego_graph_nodes[2]
            + ego_graph_nodes[3]
        )
    else:
        pretrain_ego_graph_nodes = (
            ego_graph_nodes[0] + ego_graph_nodes[1] + ego_graph_nodes[2]
        )
    ego_graph_nodes = (
        ego_graph_nodes[0] + ego_graph_nodes[1] + ego_graph_nodes[2]
    )  # * merge train/val/test = all

    logging.info(
        f"After loading data, occupied memory: {show_occupied_memory():.2f} MB"
    )  # in MB

    args.num_features = feats.shape[1]

    if logs:
        logger = WandbLogger(
            log_path=f"{dataset_name}_loss_{loss_fn}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}",
            project="GraphMAE2",
            args=args,
        )
    else:
        logger = None
    model_name = f"{encoder}_{decoder}_{num_hidden}_{num_layers}_{dataset_name}_{args.mask_rate}_{num_hidden}_checkpoint.pt"

    model = build_model(args)

    if not args.no_pretrain:
        # ------------- pretraining starts ----------------
        if not load_model:
            logging.info("---- start pretraining ----")
            model = pretrain(
                model,
                feats,
                graph,
                pretrain_ego_graph_nodes,
                max_epoch=max_epoch,
                device=device,
                use_scheduler=use_scheduler,
                lr=lr,
                weight_decay=weight_decay,
                batch_size=batch_size,
                drop_edge_rate=drop_edge_rate,
                sampling_method=sampling_method,
                optimizer=optimizer_type,
            )

            model = model.cpu()
            logging.info(f"saving model to {model_dir}/{model_name}...")
            torch.save(model.state_dict(), os.path.join(model_dir, model_name))
        # ------------- pretraining ends ----------------

        if load_model:
            model.load_state_dict(torch.load(os.path.join(args.checkpoint_path)))
            logging.info(f"Loading Model from {args.checkpoint_path}...")
    else:
        logging.info("--- no pretrain ---")
