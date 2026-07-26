import pandas as pd
from pathlib import Path

merged_dir = Path("data/merged")

files = [
    "merged_dataset",
    "train",
    "validation",
    "test",
]

for file in files:
    parquet_path = merged_dir / f"{file}.parquet"
    csv_path = merged_dir / f"{file}.csv"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        df.to_csv(csv_path, index=False)
        print(f"✓ Converted {parquet_path.name} → {csv_path.name} ({len(df)} rows)")
    else:
        print(f"✗ {parquet_path.name} not found")

print("\nAll available Parquet files have been converted to CSV.")