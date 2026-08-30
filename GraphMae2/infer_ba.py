import torch
from datasets.lc_sampler import (
    load_dataset,
    setup_finetune_dataloder
)
from utils import build_args
from models import build_model

def load_infer_components(device="cpu"):
    data_dir = 'workspace/data/pretrain'
    pretrain_path = 'checkpoints/gat_gat_512_2_social_0.5_512_checkpoint.pt'
    dataset_name = 'social'

    feats, graph, labels, split_idx, id_mapping = load_dataset(
        data_dir, dataset_name, create_ego_graphs=False
    )

    args = build_args()
    args.num_features = feats.shape[1]
    model = build_model(args)

    print("🟢 Loading model checkpoint...")
    model.load_state_dict(torch.load(pretrain_path, map_location="cpu"))  # safer
    model = model.get_encoder()
    model.to(device)
    model.eval()

    ego_graphs_file_path = f"lc_ego_graphs/{dataset_name}/graphs.pt"
    ego_graph_nodes = torch.load(ego_graphs_file_path, map_location="cpu")
    train_egs, val_egs, test_egs = ego_graph_nodes

    batch_size = 32
    data_loader = setup_finetune_dataloder(
        "lc", graph, feats, train_egs, labels, batch_size=batch_size, shuffle=False
    )

    return id_mapping, feats, graph, labels, split_idx, model, data_loader
