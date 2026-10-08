# Report notes (running log, to turn into README.md later)

## Task 1: Application and dataset
- Application: blood cell classification from microscope images (8 classes; not only white blood cells, since erythroblast and platelet are included).
- Source: Hugging Face `Falah/Blood_8_classes_Dataset` (no license listed there). Per-class counts match the PBC dataset exactly, so it is very likely a re-upload of it:
  Acevedo et al., "A dataset of microscopic peripheral blood cell images for development of automatic recognition systems", Data in Brief, 2020 (CC BY 4.0).
- Downloaded: 17,092 images, ~302 MB, JPEG, mostly 360x363 px (a few 366x369 / 360x360).
- Counts after dedup (17,074 total):

  | class | images |
  |---|---|
  | basophil | 1212 |
  | eosinophil | 3110 |
  | erythroblast | 1551 |
  | ig | 2892 |
  | lymphocyte | 1214 |
  | monocyte | 1419 |
  | neutrophil | 3328 |
  | platelet | 2348 |

- Imbalance: largest/smallest ~2.7x (mild). Handled with a stratified split; class weights optional.
- Figures: `images/blood_class_counts.png`, `images/blood_samples.png` (3 random per class, seed 11).
- Visual inspection: each image is centered on one cell with red blood cells in the background. Stain color varies (pink to purple-blue). Some images are less sharp. Some contain extra cells (e.g. a platelet image that also shows a neutrophil), which is a form of label noise.
- Expected (to check against confusion matrix): easy = eosinophil (orange-red granules), platelet (very small). Hard = lymphocyte vs erythroblast (small, dark, round), ig vs monocyte (large, irregular nucleus).
- Implication for preprocessing: color is informative, so keep RGB and no color augmentation. Cells have no orientation, so vertical flip and full rotation are valid augmentations (unlike cats).

## Task 2: Preprocessing
- Deduplication (by file hash): removed 18 images. 16 exact duplicates within the same class, plus both copies of one image labeled both eosinophil and neutrophil (conflicting label). Reason: prevent train/test leakage and contradictory training signal.

- Split: stratified 70/15/15 on file paths (seed 11): 11,951 / 2,561 / 2,562; no overlap; saved to `results/blood_split.json` so notebook and script use the same split.
- Resize 360x363 -> 128x128 (bilinear); <1% aspect distortion; nucleus texture still visible (`images/blood_before_after_resize.png`).
- Normalize 0-255 -> 0-1 inside the tf.data loader (`scripts/blood_data.py`), so it cannot be applied twice (cats bug). Images loaded lazily in batches and cached as uint8 (~700 MB) instead of one 3.4 GB float array.
- Augmentation (training only, random each epoch): RandomFlip horizontal+vertical, RandomRotation(0.5 = ±180°), RandomZoom(0.1), fill_mode="reflect". Change vs cats: vertical flip and full rotation, because cells have no orientation. With 12k images augmentation is optional (decision: keep it; open to revisit). Preview: `images/blood_augmentation_examples.png`.
- Not the same as the rejected pre-augmented Kaggle set: our augmentation never reaches val/test and creates no stored copies.

## Task 3: Adapted CNN (`scripts/blood_model.py`)
- Input(128,128,3) -> augmentation -> 4x [Conv2D(32/64/128/128, 3x3, relu) + MaxPool] -> Flatten (4608) -> Dropout(0.3) -> Dense(64, relu) -> Dense(8) logits.
- 536,328 params; the biggest share is Dense(64): 294,976 (55%).
- Changes vs TF CIFAR-10 tutorial (3 conv layers 32/64/64, 2 MaxPools, Dense(64), Dense(10), 32x32 input): input 128x128; 4 conv layers (32/64/128/128) each followed by MaxPool, so the 128x128 maps shrink to 6x6 before Flatten (otherwise the Dense layer would be huge); augmentation layers; Dropout; Dense(8) instead of Dense(10).
- Changes vs cats model: Dense(8) instead of 7 (+65 params), Dropout 0.3 instead of 0.5.

## Task 4/5: Training run blood_run1 (`results/history_blood_run1.json`, log `results/blood_run1_log.txt`)
- Script `scripts/train_blood.py` (terminal + caffeinate, history saved each epoch, best checkpoint by val_loss, EarlyStopping patience 15). Adam lr=0.001, Dropout 0.3, batch 32, augmentation on, seed 11.
- ~69 s/epoch on M2 CPU; stopped at epoch 51 (max 60) after 63 min; best val_loss 0.126 at epoch 36 (val acc 95.9%, train acc 97.3%) -> that checkpoint is the final model.
- Epoch 1: train 57.2% / val 71.9%. Epoch 11: 94.2% / 90.3%. Epoch 21: 95.9% / 91.4%. Epoch 51: 97.7% / 95.8% (max val acc 96.3% at epoch 50).
- Convergence: fast rise in first ~10 epochs, then slow improvement, plateau from ~epoch 36 (val loss 0.13-0.15 for the remaining 15 epochs while train loss kept falling slowly 0.077 -> 0.067). Likely converged; small train-val gap (1-2 points) = little overfitting.
- Val loss noisier than train loss, but far steadier than cats (2,561 val images: 1 image = 0.04 points vs 2.6 for cats).
- Contrast with cats: same architecture went from val ~36% to ~96%. Likely main reason: data size/quality (12k vs 179 training images), plus visually distinct, centered, uniformly imaged classes.

## Why the dataset changed (cats -> blood cells)
- Preliminary experiment (git tag `cats-experiments`, files in `archive-cats/`): 7 cat species, 257 images. A from-scratch CNN overfit (train ~78%, val ~36%); 39 validation images made val metrics very noisy (1 image = 2.6 points).
- Lesson carried over: Dropout(0.5) on 179 training images likely blocked learning (Run 1; single run, hypothesis).
- Rejected alternative: Kaggle "Blood Cell Images" (4 classes); its 12,500 images are augmented copies of ~410 originals, so same small-data problem plus leakage.
