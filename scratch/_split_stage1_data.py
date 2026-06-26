"""Split R2AIStage1DATA.json into 4 equal parts."""
import json
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "data" / "stage6_data" / "R2AIStage1DATA.json"
OUT_DIR = SRC.parent

with SRC.open("r", encoding="utf-8") as f:
    data = json.load(f)

total = len(data)
n_parts = 4
base = total // n_parts
remainder = total % n_parts

print(f"Source: {SRC}")
print(f"Total questions: {total}")

start = 0
for i in range(n_parts):
    # Distribute the remainder across the first `remainder` parts
    size = base + (1 if i < remainder else 0)
    chunk = data[start:start + size]
    out_path = OUT_DIR / f"R2AIStage1DATA_part{i + 1}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(chunk, f, ensure_ascii=False, indent=2)
        f.write("\n")
    ids = [q["id"] for q in chunk]
    print(
        f"  part{i + 1}: {len(chunk)} questions, "
        f"id range {ids[0]}-{ids[-1]} -> {out_path.name}"
    )
    start += size

print("Done.")
