#!/usr/bin/env python3
import os
import gc
import numpy as np
import pandas as pd

def main():
    # 1. Load orig_index
    parquet_path = 'workshop/graph_feats/filtered_sampled_twibot_allcols_with_index.parquet'
    print(f"🔍 Loading orig_index from {parquet_path}")
    df = pd.read_parquet(parquet_path, columns=['orig_index'])
    orig_idx = df['orig_index'].values  # shape = (N,)

    # 2. Prepare a boolean array to track matches
    matched = np.zeros(orig_idx.shape[0], dtype=bool)

    # 3. Iterate over all chunks and accumulate marks
    cum_offset = 0
    feat_dir = 'workshop/graph_feats'
    print(f"🔄 Checking feature files in {feat_dir}")

    for i in range(9):
        feat_file = os.path.join(feat_dir, f'feat_{i}.npy')
        if not os.path.exists(feat_file):
            print(f"  ⚠️ Skipping missing file: {feat_file}")
            continue

        feat = np.load(feat_file, mmap_mode='r')
        Ni = feat.shape[0]
        start, end = cum_offset, cum_offset + Ni

        mask = (orig_idx >= start) & (orig_idx < end)
        matched[mask] = True  # mark as matched

        # cleanup
        del feat, mask
        gc.collect()
        cum_offset += Ni

        print(f"  ▪ Chunk {i}: rows={Ni}, global_idx_range=[{start},{end})")

    # 4. Report results
    total = orig_idx.shape[0]
    matched_count = matched.sum()
    print("\n📊 Match statistics:")
    print(f"  • Total orig_index: {total}")
    print(f"  • Number matched: {matched_count}")
    print(f"  • Number unmatched: {total - matched_count}")

    if matched_count < total:
        print("  ❌ Example of unmatched indices:", orig_idx[~matched][:20])
    else:
        print("  ✅ All orig_index entries were successfully matched!")


if __name__ == "__main__":
    main()
