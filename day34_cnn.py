# DAY 34: Convolutional Neural Networks
# Image Classification with CNN 🖼️

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist
from sklearn.metrics import classification_report

print(f"TensorFlow: {tf.__version__}")
print("CNN - Convolutional Neural Networks! 🖼️")

# Load MNIST
print("\n===== LOADING MNIST DATASET =====")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f"Training images: {X_train.shape}")
print(f"Testing images: {X_test.shape}")
print(f"Image size: {X_train[0].shape}")
print(f"Pixel range: {X_train.min()} to {X_train.max()}")
print(f"Classes: {np.unique(y_train)}")

# Sample images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i], cmap="gray")
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis("off")
plt.suptitle("Sample MNIST Images")
plt.tight_layout()
plt.show()

# Preprocessing
print("\n===== PREPROCESSING =====")
X_train = X_train / 255.0
X_test = X_test / 255.0

print(f"After normalization:")
print(f"Min: {X_train.min():.1f}")
print(f"Max: {X_train.max():.1f}")

X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)

print(f"\nCNN input shape: {X_train_cnn.shape}")
y_train_oh = keras.utils.to_categorical(y_train, 10)
y_test_oh = keras.utils.to_categorical(y_test, 10)
print(f"Label shape: {y_train_oh.shape}")

# Model 1: Dense NN
print("\n===== MODEL 1: DENSE NN =====")
model_dense = keras.Sequential([
    keras.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model_dense.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model_dense.summary()

history_dense = model_dense.fit(
    X_train, y_train_oh,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

dense_loss, dense_acc = model_dense.evaluate(
    X_test, y_test_oh, verbose=0
)
print(f"\nDense NN Accuracy: {dense_acc*100:.2f}%")

# Model 2: CNN
print("\n===== MODEL 2: CNN =====")
model_cnn = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),

    # Conv Block 1
    layers.Conv2D(32, (3,3),
                  activation="relu",
                  padding="same"),
    layers.Conv2D(32, (3,3),
                  activation="relu",
                  padding="same"),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Conv Block 2
    layers.Conv2D(64, (3,3),
                  activation="relu",
                  padding="same"),
    layers.Conv2D(64, (3,3),
                  activation="relu",
                  padding="same"),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Classifier
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])

model_cnn.summary()

model_cnn.compile(
    optimizer=keras.optimizers.Adam(
              learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

print("\nTraining CNN...")
history_cnn = model_cnn.fit(
    X_train_cnn, y_train_oh,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

cnn_loss, cnn_acc = model_cnn.evaluate(
    X_test_cnn, y_test_oh, verbose=0
)
print(f"\nCNN Accuracy: {cnn_acc*100:.2f}%")

# Comparison Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history_dense.history["accuracy"],
             label="Dense Train")
axes[0].plot(history_dense.history["val_accuracy"],
             label="Dense Val")
axes[0].plot(history_cnn.history["accuracy"],
             label="CNN Train")
axes[0].plot(history_cnn.history["val_accuracy"],
             label="CNN Val")
axes[0].set_title("Dense vs CNN Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(history_dense.history["loss"],
             label="Dense")
axes[1].plot(history_cnn.history["loss"],
             label="CNN")
axes[1].set_title("Dense vs CNN Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.show()

# Predictions
print("\n===== PREDICTIONS =====")
y_pred_proba = model_cnn.predict(X_test_cnn)
y_pred = np.argmax(y_pred_proba, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize predictions
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i], cmap="gray")
    pred = y_pred[i]
    true = y_test[i]
    color = "green" if pred==true else "red"
    ax.set_title(f"True:{true} Pred:{pred}",
                 color=color)
    ax.axis("off")
plt.suptitle("CNN Predictions (Green=Correct Red=Wrong)")
plt.tight_layout()
plt.show()

# Wrong predictions
print("\n===== WRONG PREDICTIONS =====")
wrong_idx = np.where(y_pred != y_test)[0]
print(f"Total wrong: {len(wrong_idx)} / {len(y_test)}")
print(f"Accuracy: {(1-len(wrong_idx)/len(y_test))*100:.2f}%")

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    if i < len(wrong_idx):
        idx = wrong_idx[i]
        ax.imshow(X_test[idx], cmap="gray")
        ax.set_title(
            f"True:{y_test[idx]}\nPred:{y_pred[idx]}",
            color="red")
        ax.axis("off")
plt.suptitle("Wrong Predictions")
plt.tight_layout()
plt.show()

# Feature Maps (Fixed!)
print("\n===== FEATURE MAPS =====")
feature_model = keras.Model(
    inputs=model_cnn.inputs,     # ← fixed!
    outputs=model_cnn.layers[0].output
)

sample_image = X_test_cnn[0:1]
feature_maps = feature_model.predict(
    sample_image, verbose=0)

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for i, ax in enumerate(axes.flat):
    if i < 32:
        ax.imshow(feature_maps[0,:,:,i],
                  cmap="viridis")
        ax.axis("off")
plt.suptitle("Feature Maps - Conv Layer 1 (32 filters)")
plt.tight_layout()
plt.show()

# Final Comparison
print("\n===== FINAL COMPARISON =====")
print(f"{'Model':15} {'Accuracy':10} {'Params':10}")
print("-" * 37)

dense_params = model_dense.count_params()
cnn_params = model_cnn.count_params()

print(f"{'Dense NN':15} {dense_acc*100:8.2f}%  "
      f"{dense_params:,}")
print(f"{'CNN':15} {cnn_acc*100:8.2f}%  "
      f"{cnn_params:,}")

winner = "CNN 🖼️" if cnn_acc > dense_acc \
         else "Dense NN"
print(f"\n🏆 Winner: {winner}")
print(f"CNN improvement: "
      f"{(cnn_acc-dense_acc)*100:.2f}%")

# Summary
print("\n===== DAY 34 SUMMARY =====")
print(f"Dataset: MNIST")
print(f"Training: 60,000 images")
print(f"Testing: 10,000 images")
print(f"Image: 28×28 pixels")
print(f"Classes: 10 digits (0-9)")
print(f"\nDense NN: {dense_acc*100:.2f}%")
print(f"CNN:      {cnn_acc*100:.2f}%")
print(f"Wrong: {len(wrong_idx)}/10,000")
print(f"\n🖼️ CNN — COMPLETE!")