import pandas as pd

def main():
    # 1. Path setup
    in_path = "workshop/graph_feats/source_twitbot.parquet"
    author_ids_path = "workshop/graph_feats/author_ids_combined.csv"
    out_path = "workshop/graph_feats/source_twitbot_filtered.parquet"

    # 2. Read the original Parquet file
    df = pd.read_parquet(in_path)

    # 3. Read author_ids, keep them as strings
    author_ids_df = pd.read_csv(author_ids_path)
    author_id_set = set(author_ids_df["id"].astype(str))

    # 4. Filter rows where author_id is in the set
    df_filtered = df[df["author_id"].astype(str).isin(author_id_set)].reset_index(drop=True)

    # 5. Save the result
    df_filtered.to_parquet(out_path, index=False)

    # 6. Print info
    print(f"Original row count: {len(df)}, Filtered row count: {len(df_filtered)}")
    print(f"Filtered result saved to: {out_path}")

if __name__ == "__main__":
    main()

