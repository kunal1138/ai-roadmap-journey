# DAY 33: Advanced Neural Networks
# Dropout, BatchNorm, Early Stopping 🧠

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

print(f"TensorFlow: {tf.__version__}")
print("Advanced Neural Networks! 🚀")

# Dataset
np.random.seed(42)
n = 1000

data = {
    "Study_Hours": np.random.uniform(1, 10, n),
    "Attendance": np.random.uniform(50, 100, n),
    "Maths": np.random.randint(30, 100, n),
    "Science": np.random.randint(30, 100, n),
    "English": np.random.randint(30, 100, n),
    "Programming": np.random.randint(30, 100, n),
    "Projects_Done": np.random.randint(0, 10, n),
    "Certifications": np.random.randint(0, 5, n),
}

df = pd.DataFrame(data)
df["Percentage"] = (
    (df["Maths"] + df["Science"] +
     df["English"] + df["Programming"])
    / 400 * 100
).round(2)

def get_performance(row):
    if (row["Percentage"] >= 75 and
        row["Study_Hours"] >= 6 and
        row["Attendance"] >= 80):
        return "Excellent"
    elif (row["Percentage"] >= 55 and
          row["Study_Hours"] >= 4):
        return "Good"
    elif row["Percentage"] >= 40:
        return "Average"
    else:
        return "Poor"

df["Performance"] = df.apply(get_performance, axis=1)
print(f"\nDataset: {df.shape}")
print(df["Performance"].value_counts())

# Preprocessing
features = ["Study_Hours","Attendance","Maths",
            "Science","English","Programming",
            "Projects_Done","Certifications"]

X = df[features].values
y = df["Performance"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_onehot = keras.utils.to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining: {len(X_train)}")
print(f"Testing: {len(X_test)}")

# Model 1: Basic
print("\n===== MODEL 1: BASIC NEURAL NETWORK =====")
model_basic = keras.Sequential([
    keras.Input(shape=(8,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(4, activation="softmax")
])

model_basic.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_basic = model_basic.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

basic_loss, basic_acc = model_basic.evaluate(
    X_test_scaled, y_test, verbose=0
)
print(f"Basic Model Accuracy: {basic_acc*100:.2f}%")

final_train = history_basic.history["accuracy"][-1]
final_val = history_basic.history["val_accuracy"][-1]
gap = (final_train - final_val) * 100
print(f"Train Accuracy: {final_train*100:.2f}%")
print(f"Val Accuracy: {final_val*100:.2f}%")
print(f"Overfitting Gap: {gap:.2f}%")

# Model 2: Advanced
print("\n===== MODEL 2: ADVANCED NEURAL NETWORK =====")
print("(With Dropout + Batch Normalization)")

model_advanced = keras.Sequential([
    layers.Dense(128, input_shape=(8,)),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(0.3),

    layers.Dense(64),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(0.3),

    layers.Dense(32),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(0.2),

    layers.Dense(4, activation="softmax")
])

model_advanced.summary()

model_advanced.compile(
    optimizer=keras.optimizers.Adam(
              learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
print("\n===== CALLBACKS =====")
early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=0.0001,
    verbose=1
)

history_advanced = model_advanced.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

adv_loss, adv_acc = model_advanced.evaluate(
    X_test_scaled, y_test, verbose=0
)
print(f"\nAdvanced Model Accuracy: {adv_acc*100:.2f}%")

final_train_adv = history_advanced.history["accuracy"][-1]
final_val_adv = history_advanced.history["val_accuracy"][-1]
gap_adv = (final_train_adv - final_val_adv) * 100
print(f"Train Accuracy: {final_train_adv*100:.2f}%")
print(f"Val Accuracy: {final_val_adv*100:.2f}%")
print(f"Overfitting Gap: {gap_adv:.2f}%")

# Comparison Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].plot(history_basic.history["accuracy"],
               label="Train", color="blue")
axes[0,0].plot(history_basic.history["val_accuracy"],
               label="Validation", color="orange")
axes[0,0].set_title("Basic Model - Accuracy")
axes[0,0].set_xlabel("Epoch")
axes[0,0].set_ylabel("Accuracy")
axes[0,0].legend()

axes[0,1].plot(history_basic.history["loss"],
               label="Train", color="blue")
axes[0,1].plot(history_basic.history["val_loss"],
               label="Validation", color="orange")
axes[0,1].set_title("Basic Model - Loss")
axes[0,1].set_xlabel("Epoch")
axes[0,1].set_ylabel("Loss")
axes[0,1].legend()

axes[1,0].plot(
    history_advanced.history["accuracy"],
    label="Train", color="green")
axes[1,0].plot(
    history_advanced.history["val_accuracy"],
    label="Validation", color="red")
axes[1,0].set_title("Advanced Model - Accuracy")
axes[1,0].set_xlabel("Epoch")
axes[1,0].set_ylabel("Accuracy")
axes[1,0].legend()

axes[1,1].plot(
    history_advanced.history["loss"],
    label="Train", color="green")
axes[1,1].plot(
    history_advanced.history["val_loss"],
    label="Validation", color="red")
axes[1,1].set_title("Advanced Model - Loss")
axes[1,1].set_xlabel("Epoch")
axes[1,1].set_ylabel("Loss")
axes[1,1].legend()

plt.tight_layout()
plt.show()

# Final Comparison
print("\n===== MODEL COMPARISON =====")
print(f"{'Model':20} {'Accuracy':10} {'Gap':10}")
print("-" * 42)
print(f"{'Basic NN':20} {basic_acc*100:8.2f}%  "
      f"{gap:8.2f}%")
print(f"{'Advanced NN':20} {adv_acc*100:8.2f}%  "
      f"{gap_adv:8.2f}%")

winner = "Advanced NN 🧠" if adv_acc > basic_acc \
         else "Basic NN"
print(f"\n🏆 Winner: {winner}")
print(f"Overfitting reduced by: "
      f"{gap-gap_adv:.2f}%")

# Predictions
print("\n===== PREDICTIONS (Advanced Model) =====")
new_students = np.array([
    [8, 90, 90, 85, 88, 95, 5, 2],
    [2, 60, 45, 40, 50, 35, 1, 0],
    [5, 75, 65, 60, 62, 58, 3, 1],
    [7, 85, 80, 75, 78, 82, 4, 3],
])

new_scaled = scaler.transform(new_students)
predictions = model_advanced.predict(new_scaled)

names = ["Kunal C","Weak Student",
         "Average Joe","Good Student"]
print(f"\n{'Student':15} {'Performance':12} "
      f"{'Confidence':10}")
print("-" * 40)
for name, pred in zip(names, predictions):
    idx = np.argmax(pred)
    conf = pred[idx] * 100
    perf = le.classes_[idx]
    emoji = "🌟" if perf=="Excellent" else \
            "😊" if perf=="Good" else \
            "😐" if perf=="Average" else "😔"
    print(f"{name:15} {perf:12} {conf:6.2f}% {emoji}")

# Summary
print("\n===== DAY 33 SUMMARY =====")
print(f"Dataset: {n} students")
print(f"\nBasic NN:")
print(f"  Accuracy: {basic_acc*100:.2f}%")
print(f"  Overfit Gap: {gap:.2f}%")
print(f"\nAdvanced NN:")
print(f"  Accuracy: {adv_acc*100:.2f}%")
print(f"  Overfit Gap: {gap_adv:.2f}%")
print(f"\nTechniques used:")
print(f"  ✅ Dropout (0.3, 0.2)")
print(f"  ✅ Batch Normalization")
print(f"  ✅ Early Stopping")
print(f"  ✅ Learning Rate Reduction")
print(f"\n🧠 Advanced Neural Networks — COMPLETE!")