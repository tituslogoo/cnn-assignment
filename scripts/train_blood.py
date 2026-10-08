# Train the blood cell CNN from the terminal (survives VS Code / kernel crashes).
# Usage (from the project root):
#   caffeinate -i python scripts/train_blood.py --run blood_run1 --epochs 60
import argparse
import json
import time

from tensorflow import keras

from blood_data import ROOT, SEED, load_split, make_dataset
from blood_model import build_model

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True, help="prefix for output files, e.g. blood_run1")
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--patience", type=int, default=15)
args = parser.parse_args()

results_dir = ROOT / "results"
history_file = results_dir / f"history_{args.run}.json"
best_model_file = results_dir / f"{args.run}_best.keras"
if history_file.exists():
    raise SystemExit(f"{history_file} already exists; pick a new --run name (never overwrite old runs)")

keras.utils.set_random_seed(SEED)
split = load_split()
train_ds = make_dataset(split, "train", shuffle=True)
val_ds = make_dataset(split, "val")
model = build_model(dropout=args.dropout, lr=args.lr)


class SaveHistoryEachEpoch(keras.callbacks.Callback):
    """Write the history JSON after every epoch, so a crash still leaves results."""

    def __init__(self):
        super().__init__()
        self.history = {}
        self.start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        for k, v in (logs or {}).items():
            self.history.setdefault(k, []).append(float(v))
        self.history.setdefault("elapsed_s", []).append(round(time.time() - self.start, 1))
        with open(history_file, "w") as f:
            json.dump({"args": vars(args), "history": self.history}, f, indent=1)


callbacks = [
    SaveHistoryEachEpoch(),
    keras.callbacks.ModelCheckpoint(best_model_file, monitor="val_loss", save_best_only=True),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience),
]

print(f"Run {args.run}: {vars(args)}")
model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=2)
print(f"Done. History: {history_file}  Best model: {best_model_file}")
