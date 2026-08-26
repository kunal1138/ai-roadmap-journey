# DAY 25: Support Vector Machine (SVM)
# Gaming Rank Prediction with Different Kernels

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Gaming dataset
data = {
    "Level": [12,23,24,67,78,87,88,99,
              15,25,35,45,55,65,75,85],
    "Wins": [232,234,223,123,342,345,234,345,
             150,200,220,180,300,320,280,340],
    "Loses": [123,454,453,345,345,344,456,432,
              200,300,250,400,200,150,300,250],
    "Hours": [2,3,4,5,6,7,8,9,
              1,2,3,4,5,6,7,8],
    "Rank": ["Noob","Noob","Noob","Pro",
             "Pro","Legend","Pro","Legend",
             "Noob","Noob","Noob","Noob",
             "Pro","Legend","Pro","Legend"]
}

df = pd.DataFrame(data)
print(df)

# Features and Labels
X = df[["Level","Wins","Loses","Hours"]]
y = df["Rank"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features - important for SVM!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Test all kernels
kernels = ["linear", "rbf", "poly"]
for kernel in kernels:
    model = SVC(kernel=kernel, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Kernel: {kernel:8} -> Accuracy: {accuracy*100:.2f}%")

# Best model - Linear kernel
print("\n===Best Model: Linear KERNEL===")
best_model = SVC(kernel="linear", random_state=42)
best_model.fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

# Classification Report
print("\n===Classification Report===")
print(classification_report(y_test, y_pred,
      zero_division=0))

# Support Vectors
print("\nSupport Vector Classes:")
for i, class_name in enumerate(best_model.classes_):
    print(f"{class_name}: {best_model.n_support_[i]} vectors")

# Algorithm Comparison
print("\n===== ALGORITHM COMPARISON =====")
algorithms = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier(max_depth=3),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(kernel="linear")
}

for name, algo in algorithms.items():
    algo.fit(X_train_scaled, y_train)
    pred = algo.predict(X_test_scaled)
    acc = accuracy_score(y_test, pred)
    print(f"{name:20}: {acc*100:.2f}%")

# Predict new player
new_player = pd.DataFrame({
    "Level": [90],
    "Wins": [380],
    "Loses": [80],
    "Hours": [9]
})
new_scaled = scaler.transform(new_player)
prediction = best_model.predict(new_scaled)
print(f"\nPlayer: Level 90, 380 Wins, 80 Loses, 9 Hours")
print(f"Rank: {prediction[0]}")