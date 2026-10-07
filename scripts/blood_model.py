# Shared augmentation and model definition for the blood cell notebook and training script
from tensorflow import keras
from tensorflow.keras import layers

SEED = 11


def make_augmentation():
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=SEED),
        layers.RandomRotation(0.5, fill_mode="reflect", seed=SEED),  # 0.5 = up to ±180°
        layers.RandomZoom(0.1, fill_mode="reflect", seed=SEED),
    ], name="augmentation")
