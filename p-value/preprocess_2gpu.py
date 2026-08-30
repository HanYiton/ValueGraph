import torch
from tqdm import tqdm
from dataset_tool import fast_merge
import numpy as np
from transformers import pipeline
import os
import pandas as pd
import json
import torch
from transformers import pipeline
from tqdm import tqdm
import multiprocessing as mp
import json





import os
import json
import torch
from transformers import pipeline
from tqdm import tqdm
import multiprocessing as mp


def worker(device_id, user_ids, each_user_tweets, output_path, max_tweets_per_user=20, batch_size=16):
    print(f"Worker on GPU {device_id} processing {len(user_ids)} users...")

    feature_extract = pipeline(
        'feature-extraction',
        model='roberta-base',
        tokenizer='roberta-base',
        device=device_id,
        batch_size=batch_size
    )

    tweets_list = []

    for i in tqdm(user_ids, desc=f"GPU {device_id}"):
        user_tweets = each_user_tweets.get(str(i), [])

        if not user_tweets:
            tweets_list.append(torch.zeros(768))
            continue

        # Limit and clean tweets
        tweets = user_tweets[:max_tweets_per_user]
        tweets = [t if isinstance(t, str) and t.strip() else "" for t in tweets]

        # Filter out empty ones
        valid_tweets = [t for t in tweets if t]

        tweet_embs = []

        with torch.no_grad():
            for b in range(0, len(valid_tweets), batch_size):
                batch = valid_tweets[b: b + batch_size]
                try:
                    batch_outputs = feature_extract(batch, truncation=True, padding=True, max_length=50)
                    for out in batch_outputs:
                        out_tensor = torch.tensor(out, dtype=torch.float32)
                        if out_tensor.ndim == 2 and out_tensor.shape[1] == 768:
                            tweet_embs.append(out_tensor.mean(dim=0))
                except Exception as e:
                    print(f"[GPU {device_id}] Error during batch processing: {e}")

        if tweet_embs:
            user_embedding = torch.stack(tweet_embs).mean(dim=0)
        else:
            user_embedding = torch.zeros(768)

        tweets_list.append(user_embedding)

    # Save partial results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(torch.stack(tweets_list), output_path)
    print(f"Worker on GPU {device_id} finished. Saved to {output_path}")


def parallel_tweet_embedding(each_user_tweets):
    total_users = len(each_user_tweets)
    user_ids = list(range(total_users))
    chunk_size = total_users // 4

    # Split into 4 parts
    chunks = [user_ids[i:i+chunk_size] for i in range(0, total_users, chunk_size)]
    devices = [0, 0, 1, 1]

    processes = []
    paths = []

    for idx, (dev, chunk) in enumerate(zip(devices, chunks)):
        path = f"./processed_data/tweets_tensor_gpu{dev}_{idx}.pt"
        paths.append(path)
        p = mp.Process(target=worker, args=(dev, chunk, each_user_tweets, path))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Merge and save
    all_parts = [torch.load(p) for p in paths]
    merged_tensor = torch.cat(all_parts, dim=0)
    final_path = "./processed_data/tweets_tensor.pt"
    torch.save(merged_tensor, final_path)
    print(f"✅ All finished. Saved merged tensor to {final_path}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    # Load JSON
    tweet_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'processed_data/id_tweet.json'))
    with open(tweet_path, 'r') as f:
        each_user_tweets = json.load(f)

    parallel_tweet_embedding(each_user_tweets)
