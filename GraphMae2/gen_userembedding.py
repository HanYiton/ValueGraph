#!/usr/bin/env python3

# import os
# import torch
# from infer_batch_new import model, graph, feats, id_mapping

# def main():
#     # 限制多线程以节省资源
#     os.environ['OMP_NUM_THREADS'] = '2'
#     os.environ['MKL_NUM_THREADS'] = '2'
    

#     # 设置设备
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 加载模型检查点
#     print("📥 Loading checkpoint...")
#     ckpt_path = "workspace/baseline/GraphMae2/our_checkpoints/best_model.pt"
#     ckpt = torch.load(ckpt_path, map_location=device,)
#     model.load_state_dict(ckpt['model_state_dict'])

#     # 准备模型和图到设备
#     model.eval()
#     model.to(device)
#     graph.to(device)

#     # 准备全量特征
#     print("🔄 Preparing full feature tensor...")
#     feats_tensor = torch.tensor(feats, dtype=torch.float32, device=device)

#     # 生成所有 post embedding
#     print("🚀 Generating post embeddings for all nodes...")
#     with torch.no_grad():
#         if hasattr(model, 'encoder'):
#             embeddings = model.encoder(graph, feats_tensor)
#         else:
#             embeddings = model(graph, feats_tensor)

#     # 保存到单一文件
#     output_dir = "workspace/data/finetune/twibot/post_embeddings"
#     os.makedirs(output_dir, exist_ok=True)
#     save_path = os.path.join(output_dir, "post_embeddings_new.pt")
#     torch.save(embeddings.cpu(), save_path)
#     print(f"✅ Saved all post embeddings ({embeddings.shape[0]} x {embeddings.shape[1]}) to {save_path}")

# if __name__ == "__main__":
#     main()
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from infer_batch_new import model, graph, feats, id_mapping
import dgl

def infer_in_batches(model, full_graph, feats_tensor, batch_size=8192):
    """
    用 mini-batch 的方式对图中的所有节点进行推理，避免显存溢出。
    """
    device = feats_tensor.device
    model.eval()
    embeddings_list = []

    node_indices = torch.arange(feats_tensor.size(0))
    dataset = TensorDataset(node_indices)
    loader = DataLoader(dataset, batch_size=batch_size)

    with torch.no_grad():
        for batch in loader:
            batch_ids = batch[0].to(device)

            # 提取子图并获取对应特征
            subgraph = dgl.node_subgraph(full_graph, batch_ids)
            sub_feats = feats_tensor[batch_ids]

            # 推理模型（必须支持子图推理）
            output = model(subgraph, sub_feats)

            # 收集结果（放回 CPU）
            embeddings_list.append(output.cpu())

    return torch.cat(embeddings_list, dim=0)

def main():
    # 限制 PyTorch 线程数以节省资源
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型权重
    print("📥 Loading checkpoint...")
    ckpt_path = "workspace/baseline/GraphMae2/our_checkpoints/best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    # 模型与图迁移到 GPU
    model.to(device)
    graph.to(device)

    # 构造特征 tensor
    print("🔄 Preparing full feature tensor...")
    feats_tensor = feats.clone().detach().to(device)

    # 执行推理
    print("🚀 Generating post embeddings for all nodes in batches...")
    embeddings = infer_in_batches(model, graph, feats_tensor, batch_size=8192)

    # 保存嵌入向量
    output_dir = "workshop/enron_spam_data"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "content_embeddings.pt")
    torch.save(embeddings, save_path)
    print(f"✅ Saved all post embeddings ({embeddings.shape[0]} x {embeddings.shape[1]}) to {save_path}")

if __name__ == "__main__":
    main()
