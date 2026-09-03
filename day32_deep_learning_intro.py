# DAY 32: Introduction to Deep Learning
# Neural Networks with TensorFlow & Keras 🧠

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

print(f"TensorFlow version: {tf.__version__}")
print("Deep Learning starts NOW! 🚀")

# Dataset
np.random.seed(42)
n = 500

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

print(f"\nClasses: {le.classes_}")
print(f"Output shape: {y_onehot.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Input features: {X_train.shape[1]}")
print(f"Output classes: {y_onehot.shape[1]}")

# Build Neural Network
print("\n===== BUILDING NEURAL NETWORK =====")
model = keras.Sequential([
    layers.Dense(64, activation="relu",
                 input_shape=(8,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(4, activation="softmax")
])
model.summary()

# Compile
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
print("\n===== TRAINING NEURAL NETWORK =====")
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate
print("\n===== MODEL EVALUATION =====")
test_loss, test_accuracy = model.evaluate(
    X_test_scaled, y_test, verbose=0
)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Training History Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["accuracy"],
             label="Train Accuracy")
axes[0].plot(history.history["val_accuracy"],
             label="Val Accuracy")
axes[0].set_title("Model Accuracy over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(history.history["loss"],
             label="Train Loss")
axes[1].plot(history.history["val_loss"],
             label="Val Loss")
axes[1].set_title("Model Loss over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.show()

# Predictions
print("\n===== PREDICTIONS =====")
y_pred_proba = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)

nn_accuracy = accuracy_score(y_true, y_pred)
print(f"Neural Network Accuracy: {nn_accuracy*100:.2f}%")

# Predict New Students
print("\n===== PREDICT NEW STUDENTS =====")
new_students = np.array([
    [8, 90, 90, 85, 88, 95, 5, 2],
    [2, 60, 45, 40, 50, 35, 1, 0],
    [5, 75, 65, 60, 62, 58, 3, 1],
])

new_scaled = scaler.transform(new_students)
predictions = model.predict(new_scaled)

names = ["Kunal C", "Weak Student", "Average Joe"]
print(f"\n{'Student':15} {'Prediction':12} {'Confidence':10}")
print("-" * 40)
for name, pred in zip(names, predictions):
    class_idx = np.argmax(pred)
    confidence = pred[class_idx] * 100
    performance = le.classes_[class_idx]
    emoji = "🌟" if performance=="Excellent" else \
            "😊" if performance=="Good" else \
            "😐" if performance=="Average" else "😔"
    print(f"{name:15} {performance:12} "
          f"{confidence:6.2f}% {emoji}")

# ML vs DL Comparison
print("\n===== ML vs DEEP LEARNING =====")
y_train_orig = np.argmax(y_train, axis=1)
y_test_orig = np.argmax(y_test, axis=1)

rf = RandomForestClassifier(n_estimators=100,
                             random_state=42)
rf.fit(X_train_scaled, y_train_orig)
rf_pred = rf.predict(X_test_scaled)
rf_acc = accuracy_score(y_test_orig, rf_pred)

print(f"Random Forest (ML): {rf_acc*100:.2f}%")
print(f"Neural Network (DL): {nn_accuracy*100:.2f}%")
winner = "Neural Network 🧠" if nn_accuracy > rf_acc \
         else "Random Forest 🌲"
print(f"Winner: {winner}")

# Summary
print("\n===== NEURAL NETWORK SUMMARY =====")
print(f"Architecture: 8→64→32→16→4")
print(f"Total layers: 4")
print(f"Activation: ReLU (hidden), Softmax (output)")
print(f"Optimizer: Adam")
print(f"Loss: Categorical Crossentropy")
print(f"Epochs trained: 50")
print(f"Final Accuracy: {test_accuracy*100:.2f}%")
print("\n🧠 Deep Learning Day 1 — COMPLETE!")