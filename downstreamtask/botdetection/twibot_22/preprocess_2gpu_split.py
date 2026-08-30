import os
import json
import torch
from tqdm import tqdm
from transformers import pipeline
import multiprocessing as mp


def split_and_save_json(each_user_tweets, save_dir="./processed_data"):
    print("[MAIN] Splitting dataset into 4 JSON files...")
    total = len(each_user_tweets)
    chunks = [list(each_user_tweets.items())[i:i + total // 4] for i in range(0, total, total // 4)]
    os.makedirs(save_dir, exist_ok=True)
    for idx, chunk in enumerate(chunks):
        path = os.path.join(save_dir, f"tweets_chunk_{idx}.json")
        with open(path, 'w') as f:
            json.dump(dict(chunk), f)
    print("[MAIN] Split done.")


def worker(device_id, chunk_path, output_path, max_tweets_per_user=20, batch_size=16):
    print(f"[GPU {device_id}] Loading data from {chunk_path}...")
    with open(chunk_path, 'r') as f:
        each_user_tweets = json.load(f)

    feature_extract = pipeline(
        'feature-extraction',
        model='roberta-base',
        tokenizer='roberta-base',
        device=device_id,
        batch_size=batch_size
    )

    tweets_list = []

    for i, (user_id, user_tweets) in enumerate(tqdm(each_user_tweets.items(), desc=f"GPU {device_id}")):
        if not user_tweets:
            tweets_list.append(torch.zeros(768))
            continue

        tweets = user_tweets[:max_tweets_per_user]
        tweets = [t if isinstance(t, str) and t.strip() else "" for t in tweets]
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
                    print(f"[GPU {device_id}] Error during batch: {e}")

        if tweet_embs:
            user_embedding = torch.stack(tweet_embs).mean(dim=0)
        else:
            user_embedding = torch.zeros(768)

        tweets_list.append(user_embedding)

    torch.save(torch.stack(tweets_list), output_path)
    print(f"[GPU {device_id}] Done. Saved to {output_path}")


def parallel_embedding_from_split():
    # GPU-Worker assignment: 0,0,1,1
    chunk_paths = [
        "./processed_data/tweets_chunk_0.json",
        "./processed_data/tweets_chunk_1.json",
        "./processed_data/tweets_chunk_2.json",
        "./processed_data/tweets_chunk_3.json"
    ]
    devices = [0, 0, 1, 1]
    output_paths = [
        "./processed_data/tweets_tensor_gpu0_0.pt",
        "./processed_data/tweets_tensor_gpu0_1.pt",
        "./processed_data/tweets_tensor_gpu1_0.pt",
        "./processed_data/tweets_tensor_gpu1_1.pt"
    ]

    processes = []
    for i in range(4):
        p = mp.Process(target=worker, args=(devices[i], chunk_paths[i], output_paths[i]))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # merge and save
    all_parts = [torch.load(path) for path in output_paths]
    merged = torch.cat(all_parts, dim=0)
    torch.save(merged, "./processed_data/tweets_tensor.pt")
    print("✅ All finished. Merged tensor saved to ./processed_data/tweets_tensor.pt")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    # Step 1: Only run once to split the dataset
    tweet_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'processed_data/id_tweet.json'))
    with open(tweet_path, 'r') as f:
        full_data = json.load(f)
    split_and_save_json(full_data)

    # Step 2: parallel embedding from chunks
    parallel_embedding_from_split()
