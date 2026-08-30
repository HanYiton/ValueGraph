import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

def get_embedding(texts, tokenizer, model, device):
    inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_dim]
    # Clean NaN/Inf
    emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
    return emb.to(torch.float32).cpu().numpy()

def main():
    PARQUET_PATH = "workshop/graph_feats/merged_inner.csv"
    OUT_NPY      = "workshop/graph_feats/Enron_content.npy"
    MODEL_NAME   = "answerdotai/ModernBERT-base"
    BATCH_SIZE   = 256

    # Read texts
    df = pd.read_csv(PARQUET_PATH, usecols=["content"])
    df["text"] = df["content"].fillna("").astype(str)
    N = len(df)

    # Initialize tokenizer and CPU model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_cpu = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False
    ).to("cpu").eval()

    # Try to initialize GPU model
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        torch.set_float32_matmul_precision('high')
        model_gpu = AutoModel.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False
        ).to("cuda").eval()
    else:
        model_gpu = None

    # Batch embedding generation
    hidden_dim = model_cpu.config.hidden_size
    embeddings = np.zeros((N, hidden_dim), dtype="float32")

    for start in range(0, N, BATCH_SIZE):
        end = min(start + BATCH_SIZE, N)
        texts = df["text"].iloc[start:end].tolist()

        # Try GPU first
        if use_gpu:
            emb = get_embedding(texts, tokenizer, model_gpu, "cuda")
            # If the whole batch is zero (possible from NaNs cleaned), fallback to CPU
            if np.allclose(emb, 0.0):
                emb = get_embedding(texts, tokenizer, model_cpu, "cpu")
        else:
            emb = get_embedding(texts, tokenizer, model_cpu, "cpu")

        embeddings[start:end] = emb
        if (start // BATCH_SIZE) % 10 == 0:
            print(f"[{end}/{N}] done")

    # Save
    np.save(OUT_NPY, embeddings)
    print("Finished! Saved mixed embeddings to:", OUT_NPY)

if __name__ == "__main__":
    main()
