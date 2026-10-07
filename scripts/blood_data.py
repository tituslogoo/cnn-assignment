# Shared data loading for the blood cell notebook and training script
import json
from pathlib import Path
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "blood"
SPLIT_FILE = ROOT / "results" / "blood_split.json"
IMG_SIZE = 128
SEED = 11


def load_split():
    with open(SPLIT_FILE) as f:
        return json.load(f)


def _load_image(path, label):
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))  # bilinear
    return tf.cast(tf.round(img), tf.uint8), label   # uint8 keeps the cache small


def _normalize(img, label):
    return tf.cast(img, tf.float32) / 255.0, label    # pixels 0-255 -> 0-1


def make_dataset(split, name, batch_size=32, shuffle=False):
    paths = [str(DATA_DIR / p) for p, _ in split[name]]
    labels = [y for _, y in split[name]]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE).cache()
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
