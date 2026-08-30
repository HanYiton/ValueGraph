import pandas as pd

# 1. 读取原始 Parquet 文件
in_path = "source_twitbot.parquet"
df = pd.read_parquet(in_path)

# 2. 删除包含任何空值的行
df_clean = df.dropna(how="any").reset_index(drop=True)

# 3. 保存到新的 Parquet 文件
out_path = "source_twitbot_no_na.parquet"
df_clean.to_parquet(out_path, index=False)

print(f"原始行数: {len(df)}, 清理后行数: {len(df_clean)}")
print(f"清理后的 DataFrame 已保存至: {out_path}")
