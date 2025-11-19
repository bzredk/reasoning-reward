import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = root / "data" / "rocstories" / "test.csv"
dst = root / "data" / "rocstories" / "test_500.csv"

print("read from:", src)

df = pd.read_csv(src)
print("total rows:", len(df))

df_head = df.head(500)
df_head.to_csv(dst, index=False)

print("saved:", dst, "rows:", len(df_head))
