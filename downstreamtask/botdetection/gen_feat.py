#!/usr/bin/env python3
import os
import gc
import numpy as np
import pandas as pd

def main():
    # 1. 加载 orig_index
    parquet_path = 'graph_feats/filtered_sampled_twibot_allcols_with_index.parquet'
    print(f"🔍 Loading orig_index from {parquet_path}")
    df = pd.read_parquet(parquet_path, columns=['orig_index'])
    orig_idx = df['orig_index'].values  # shape = (N,)

    # 2. 准备一个布尔数组来 track
    matched = np.zeros(orig_idx.shape[0], dtype=bool)

    # 3. 循环所有 chunk，累积标记
    cum_offset = 0
    feat_dir = 'graph_feats'
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
        matched[mask] = True  # 标记为已匹配

        # cleanup
        del feat, mask
        gc.collect()
        cum_offset += Ni

        print(f"  ▪ Chunk {i}: rows={Ni}, global_idx_range=[{start},{end})")

    # 4. 报告结果
    total = orig_idx.shape[0]
    matched_count = matched.sum()
    print("\n📊 匹配统计:")
    print(f"  • 总共 orig_index: {total}")
    print(f"  • 被匹配到的数量: {matched_count}")
    print(f"  • 未匹配到的数量: {total - matched_count}")

    if matched_count < total:
        print("  ❌ 未匹配到的索引示例：", orig_idx[~matched][:20])
    else:
        print("  ✅ 所有 orig_index 都已成功匹配！")


if __name__ == "__main__":
    main()
