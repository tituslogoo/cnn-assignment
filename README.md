# Blood Cell Classification with a CNN

A convolutional neural network, adapted from the [TensorFlow CNN tutorial](https://www.tensorflow.org/tutorials/images/cnn), that classifies microscope images of blood cells into 8 classes: basophil, eosinophil, erythroblast, ig (immature granulocytes), lymphocyte, monocyte, neutrophil and platelet.

The model is trained from scratch on 17,074 images and reaches **96.25% accuracy** on a held-out test set.

## Dataset

[`Falah/Blood_8_classes_Dataset`](https://huggingface.co/datasets/Falah/Blood_8_classes_Dataset) on Hugging Face. It is very likely a re-upload of the PBC dataset (Acevedo et al., *Data in Brief*, 2020, CC BY 4.0). Images are not stored in this repo; the download script fetches them.

## Repository structure

| Path | Contents |
|---|---|
| `notebooks/blood_classification.ipynb` | Dataset inspection, split, preprocessing figures, training curves, test evaluation |
| `scripts/download_blood.py` | Downloads the dataset into `data/blood/<class>/` |
| `scripts/blood_data.py` | Data loading: resize to 128×128, normalize, batch |
| `scripts/blood_model.py` | Augmentation and CNN definition |
| `scripts/train_blood.py` | Training script with per-epoch history and best-model checkpoint |
| `results/` | Train/val/test split, training history and log, test predictions |
| `images/` | Generated figures |
| `archive-cats/` | Earlier experiment on 7 cat species (too little data; kept for reference) |
| `data/` | Downloaded images (gitignored) |

## Setup

Requires Python 3.11 (tested with a miniforge conda environment on an M2 Mac, CPU only).

```bash
conda create -n cnn-assignment python=3.11
conda activate cnn-assignment
pip install tensorflow==2.21.0 datasets scikit-learn matplotlib pillow jupyter
```

## Usage

Run everything from the repository root.

**1. Download the dataset** (~300 MB):
```bash
python scripts/download_blood.py
```
The study also removed 18 duplicate images after downloading (see the notebook). The committed split in `results/blood_split.json` already excludes them.

**2. Train:**
```bash
caffeinate -i python scripts/train_blood.py --run blood_run2 --epochs 60
```
Options: `--lr` (default 0.001), `--dropout` (default 0.3), `--patience` (early stopping, default 15). Outputs go to `results/history_<run>.json` and `results/<run>_best.keras`. Existing run names are never overwritten. `caffeinate -i` keeps a Mac awake during training; leave it out on other systems.

**3. Explore and evaluate:** open `notebooks/blood_classification.ipynb`, select the `cnn-assignment` kernel, and run the cells in order.

## Results (run `blood_run1`)

- Adam, learning rate 0.001, batch size 32, about 69 s per epoch on CPU
- Best validation loss at epoch 36; early stopping at epoch 51
- Test accuracy 96.25% on 2,562 images
- Easiest classes: platelet and eosinophil. Hardest: monocyte, often confused with ig
