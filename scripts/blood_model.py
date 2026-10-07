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

NUM_CLASSES = 8


def build_model(img_size=128, dropout=0.3, lr=0.001):
    model = keras.Sequential([
        keras.Input(shape=(img_size, img_size, 3)),
        make_augmentation(),  # active only during training
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(dropout),
        layers.Dense(64, activation="relu"),
        layers.Dense(NUM_CLASSES),  # logits, one per class
    ], name="blood_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model
