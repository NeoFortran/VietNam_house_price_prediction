import glob
import pandas as pd

files = sorted(glob.glob(
    r"D:\dataset_machine_learning\.cache\huggingface\hub"
    r"\datasets--tinixai--vietnam-real-estates\snapshots\*\shard_*.parquet"
))

cols = ["property_type_name", "province_name", "district_name", "price", "area",
        "bedroom_count", "bathroom_count", "house_direction", "published_at"]

df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
df["price"] = pd.to_numeric(df["price"], errors="coerce")

print(df.shape)
print(df.sample(10, random_state=0))
print(df["property_type_name"].value_counts())
print((df.isna().mean() * 100).round(1))