# DAY 35: Phase 3 Final Project
# Complete Deep Learning System 🧠

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

print("=" * 55)
print("  COMPLETE DEEP LEARNING SYSTEM 🧠")
print("=" * 55)
print(f"TensorFlow: {tf.__version__}")

# ============================================
# SECTION 1: NEURAL NETWORK
# ============================================
print("\n" + "="*55)
print("SECTION 1: STUDENT PERFORMANCE")
print("="*55)

np.random.seed(42)
n = 2000

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

df["Performance"] = df.apply(
    get_performance, axis=1)
print(f"Dataset: {df.shape}")
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
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Neural Network
nn_model = keras.Sequential([
    keras.Input(shape=(8,)),
    layers.Dense(256),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(0.3),

    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(0.3),

    layers.Dense(64),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(0.2),

    layers.Dense(32),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(0.2),

    layers.Dense(4, activation="softmax")
])

nn_model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

es = EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    restore_best_weights=True,
    verbose=0
)
rlr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    verbose=0
)

print("\nTraining Neural Network...")
nn_history = nn_model.fit(
    X_train_s, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=[es, rlr],
    verbose=0
)

nn_loss, nn_acc = nn_model.evaluate(
    X_test_s, y_test, verbose=0
)
print(f"Neural Network Accuracy: {nn_acc*100:.2f}%")

# Compare with RF
y_train_orig = np.argmax(y_train, axis=1)
y_test_orig = np.argmax(y_test, axis=1)
rf = RandomForestClassifier(n_estimators=100,
                             random_state=42)
rf.fit(X_train_s, y_train_orig)
rf_acc = accuracy_score(y_test_orig,
          rf.predict(X_test_s))
print(f"Random Forest Accuracy: {rf_acc*100:.2f}%")

nn_winner = "Neural Network 🧠" \
            if nn_acc > rf_acc \
            else "Random Forest 🌲"
print(f"Winner: {nn_winner}")

# ============================================
# SECTION 2: CNN
# ============================================
print("\n" + "="*55)
print("SECTION 2: DIGIT RECOGNITION (CNN)")
print("="*55)

(X_img_train, y_img_train), \
(X_img_test, y_img_test) = mnist.load_data()

X_img_train = X_img_train / 255.0
X_img_test = X_img_test / 255.0
X_img_train = X_img_train.reshape(-1, 28, 28, 1)
X_img_test = X_img_test.reshape(-1, 28, 28, 1)
y_img_train_oh = keras.utils.to_categorical(
                  y_img_train, 10)
y_img_test_oh = keras.utils.to_categorical(
                 y_img_test, 10)

print(f"Images: {X_img_train.shape}")

cnn_model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3,3),
                  activation="relu",
                  padding="same"),
    layers.Conv2D(32, (3,3),
                  activation="relu",
                  padding="same"),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3,3),
                  activation="relu",
                  padding="same"),
    layers.Conv2D(64, (3,3),
                  activation="relu",
                  padding="same"),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])

cnn_model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

cnn_es = EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

print("\nTraining CNN...")
cnn_history = cnn_model.fit(
    X_img_train, y_img_train_oh,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    callbacks=[cnn_es],
    verbose=1
)

cnn_loss, cnn_acc = cnn_model.evaluate(
    X_img_test, y_img_test_oh, verbose=0
)
print(f"\nCNN Accuracy: {cnn_acc*100:.2f}%")

y_img_pred = np.argmax(
    cnn_model.predict(X_img_test, verbose=0),
    axis=1)
wrong = np.sum(y_img_pred != y_img_test)
print(f"Wrong: {wrong}/10,000")
print(f"Correct: {10000-wrong}/10,000")

# ============================================
# SECTION 3: VISUALIZATIONS
# ============================================
print("\n" + "="*55)
print("SECTION 3: VISUALIZATIONS")
print("="*55)

fig = plt.figure(figsize=(16, 12))

ax1 = fig.add_subplot(2, 3, 1)
ax1.plot(nn_history.history["accuracy"],
         label="Train", color="blue")
ax1.plot(nn_history.history["val_accuracy"],
         label="Val", color="orange")
ax1.set_title("Neural Network Training")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()

ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(cnn_history.history["accuracy"],
         label="Train", color="green")
ax2.plot(cnn_history.history["val_accuracy"],
         label="Val", color="red")
ax2.set_title("CNN Training")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()

ax3 = fig.add_subplot(2, 3, 3)
models = ["Random\nForest",
          "Neural\nNetwork", "CNN"]
accs = [rf_acc*100, nn_acc*100, cnn_acc*100]
colors = ["blue", "orange", "green"]
bars = ax3.bar(models, accs, color=colors)
ax3.set_title("All Models Comparison")
ax3.set_ylabel("Accuracy %")
ax3.set_ylim(0, 105)
for bar, acc in zip(bars, accs):
    ax3.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.5,
             f"{acc:.1f}%",
             ha="center", fontsize=11,
             fontweight="bold")

ax4 = fig.add_subplot(2, 3, 4)
sample_imgs = X_img_test[:5].reshape(5, 28, 28)
combined = np.hstack(sample_imgs)
ax4.imshow(combined, cmap="gray")
preds = y_img_pred[:5]
ax4.set_title(f"CNN Predictions: {list(preds)}")
ax4.axis("off")

ax5 = fig.add_subplot(2, 3, 5)
perf_counts = df["Performance"].value_counts()
ax5.pie(perf_counts.values,
        labels=perf_counts.index,
        autopct="%1.1f%%",
        colors=["gold","green","orange","red"])
ax5.set_title("Student Performance Distribution")

ax6 = fig.add_subplot(2, 3, 6)
ax6.plot(nn_history.history["loss"],
         label="Train Loss", color="blue")
ax6.plot(nn_history.history["val_loss"],
         label="Val Loss", color="orange")
ax6.set_title("Neural Network Loss")
ax6.set_xlabel("Epoch")
ax6.set_ylabel("Loss")
ax6.legend()

plt.suptitle(
    "Phase 3 Final — Deep Learning Dashboard",
    fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# ============================================
# SECTION 4: PREDICTIONS
# ============================================
print("\n" + "="*55)
print("SECTION 4: PREDICTIONS")
print("="*55)

print("\n--- Student Performance Predictions ---")
new_students = np.array([
    [8, 90, 90, 85, 88, 95, 5, 2],
    [2, 60, 45, 40, 50, 35, 1, 0],
    [5, 75, 65, 60, 62, 58, 3, 1],
    [7, 85, 80, 75, 78, 82, 4, 3],
    [9, 95, 95, 92, 90, 98, 8, 4],
])

new_scaled = scaler.transform(new_students)
nn_preds = nn_model.predict(new_scaled, verbose=0)

names = ["Kunal C","Weak Student",
         "Average Joe","Good Student",
         "Top Performer"]

print(f"\n{'Student':15} {'Performance':12} "
      f"{'Confidence':10}")
print("-" * 40)
for name, pred in zip(names, nn_preds):
    idx = np.argmax(pred)
    conf = pred[idx] * 100
    perf = le.classes_[idx]
    emoji = "🌟" if perf=="Excellent" else \
            "😊" if perf=="Good" else \
            "😐" if perf=="Average" else "😔"
    print(f"{name:15} {perf:12} "
          f"{conf:6.2f}% {emoji}")

print("\n--- Digit Recognition Predictions ---")
sample_indices = [0,1,2,3,4,5,6,7,8,9]
sample_true = y_img_test[sample_indices]
sample_preds = y_img_pred[sample_indices]

print(f"\n{'Index':8} {'True':8} "
      f"{'Predicted':12} {'Result':8}")
print("-" * 38)
for i, (true, pred) in enumerate(
    zip(sample_true, sample_preds)):
    result = "✅" if true==pred else "❌"
    print(f"{i:8} {true:8} {pred:12} {result}")

# ============================================
# SECTION 5: FINAL REPORT
# ============================================
print("\n" + "="*55)
print("SECTION 5: FINAL REPORT")
print("="*55)

print(f"""
╔══════════════════════════════════════════╗
║   PHASE 3 DEEP LEARNING REPORT 🧠       ║
╠══════════════════════════════════════════╣
║  TASK 1: Student Performance             ║
║  Dataset:     2,000 students             ║
║  Architecture: 8→256→128→64→32→4        ║
║  RF Accuracy:  {rf_acc*100:.2f}%                   ║
║  NN Accuracy:  {nn_acc*100:.2f}%                   ║
║  Winner: {nn_winner:32s}║
║                                          ║
║  TASK 2: Digit Recognition               ║
║  Dataset:     70,000 MNIST images        ║
║  Architecture: Conv→Conv→Pool×2→Dense   ║
║  CNN Accuracy: {cnn_acc*100:.2f}%                   ║
║  Wrong:  {wrong}/10,000                    ║
║  Correct:{10000-wrong}/10,000              ║
║                                          ║
║  PHASE 3 COMPLETE! 🎉                    ║
║  Ready for Phase 4: GenAI! 🚀            ║
╚══════════════════════════════════════════╝
""")

print("Deep Learning concepts mastered:")
print("  ✅ Neural Networks")
print("  ✅ Activation Functions (ReLU, Softmax)")
print("  ✅ Dropout")
print("  ✅ Batch Normalization")
print("  ✅ Early Stopping")
print("  ✅ Learning Rate Reduction")
print("  ✅ CNN - Conv2D, MaxPooling")
print("  ✅ Feature Maps")
print("  ✅ Image Classification")
print("  ✅ ML vs DL comparison")
print("\n🧠 Phase 3 Deep Learning — COMPLETE!")
print("🚀 Ready for Phase 4: Transformers & GenAI!")