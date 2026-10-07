# Download Falah/Blood_8_classes_Dataset and save images into data/blood/<class>/
## One time use so excluded from the notebooks

from pathlib import Path
from collections import Counter
from datasets import load_dataset, Image

out_dir = Path(__file__).resolve().parent.parent / "data" / "blood"

ds = load_dataset("Falah/Blood_8_classes_Dataset", split="train")
ds = ds.cast_column("image", Image(decode=False))  # keep original file bytes (no re-encoding)
class_names = ds.features["label"].names
print("Classes:", class_names)

counts = Counter()
for i, ex in enumerate(ds):
    name = class_names[ex["label"]]
    ext = Path(ex["image"]["path"] or "x.jpg").suffix or ".jpg"
    folder = out_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}_{i:05d}{ext}").write_bytes(ex["image"]["bytes"])
    counts[name] += 1

for name in class_names:
    print(f"{name:14s} {counts[name]}")
print("Total:", sum(counts.values()))