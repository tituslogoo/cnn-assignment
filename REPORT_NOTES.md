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

## Why the dataset changed (cats -> blood cells)
- Preliminary experiment (git tag `cats-experiments`, files in `archive-cats/`): 7 cat species, 257 images. A from-scratch CNN overfit (train ~78%, val ~36%); 39 validation images made val metrics very noisy (1 image = 2.6 points).
- Lesson carried over: Dropout(0.5) on 179 training images likely blocked learning (Run 1; single run, hypothesis).
- Rejected alternative: Kaggle "Blood Cell Images" (4 classes); its 12,500 images are augmented copies of ~410 originals, so same small-data problem plus leakage.
