# DAY 28: Model Evaluation & Cross Validation
# Overfitting Detection and CV Comparison

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Dataset
data = {
    "Player": ["Kunal","Sonu","Glou","Vedu",
               "Tantan","Durgesh","Om","Sam",
               "Raj","Priya","Neha","Abhinav",
               "Rohan","Vikram","Arjun","Karan"],
    "Level": [12,23,24,67,78,87,88,99,
              15,25,35,45,55,65,75,85],
    "Wins": [232,234,223,123,342,345,234,345,
             150,200,220,180,300,320,280,340],
    "Loses": [123,454,453,345,345,344,456,432,
              200,300,250,400,200,150,300,250],
    "Hours": [2,3,4,5,6,7,8,9,
              1,2,3,4,5,6,7,8],
    "Rank": ["Legend","Noob","Noob","Pro",
             "Pro","Legend","Pro","Legend",
             "Noob","Noob","Noob","Noob",
             "Pro","Legend","Pro","Legend"]
}

df = pd.DataFrame(data)

# Features and Labels
X = df[["Level","Wins","Loses","Hours"]]
y = df["Rank"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Simple Train/Test Split
print("===== SIMPLE TRAIN/TEST SPLIT =====")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

algorithms = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier(max_depth=3),
    "Random Forest": RandomForestClassifier(
                     n_estimators=100),
    "SVM": SVC(kernel="linear"),
    "Logistic Regression": LogisticRegression()
}

simple_results = {}
for name, algo in algorithms.items():
    algo.fit(X_train, y_train)
    pred = algo.predict(X_test)
    acc = accuracy_score(y_test, pred)
    simple_results[name] = acc
    print(f"{name:20}: {acc*100:.2f}%")

# 5-Fold Cross Validation
print("\n===== 5-FOLD CROSS VALIDATION =====")
kfold = KFold(n_splits=5,
              shuffle=True,
              random_state=42)

cv_results = {}
for name, algo in algorithms.items():
    scores = cross_val_score(algo, X_scaled, y,
                             cv=kfold,
                             scoring="accuracy")
    cv_results[name] = scores
    print(f"{name:20}: "
          f"Mean={scores.mean()*100:.2f}% "
          f"Std={scores.std()*100:.2f}%")

# Comparison
print("\n===== SIMPLE vs CROSS VALIDATION =====")
print(f"{'Algorithm':20} {'Simple':10} {'CV Mean':10}")
print("-" * 42)
for name in algorithms.keys():
    simple = simple_results[name]*100
    cv_mean = cv_results[name].mean()*100
    diff = cv_mean - simple
    arrow = "↑" if diff > 0 else \
            "↓" if diff < 0 else "="
    print(f"{name:20} {simple:8.2f}%  "
          f"{cv_mean:8.2f}% {arrow}")

# Overfitting Demo
print("\n===== OVERFITTING DEMONSTRATION =====")
print("Decision Tree with different depths:")
print(f"{'Max Depth':12} {'Train Acc':12} "
      f"{'Test Acc':10} {'Status'}")
print("-" * 50)

for depth in [1, 2, 3, 5, 10, None]:
    dt = DecisionTreeClassifier(max_depth=depth,
                                random_state=42)
    dt.fit(X_train, y_train)
    train_acc = accuracy_score(y_train,
                               dt.predict(X_train))
    test_acc = accuracy_score(y_test,
                              dt.predict(X_test))
    depth_str = str(depth) if depth else "None(full)"

    if train_acc - test_acc > 0.2:
        status = "⚠️ Overfitting!"
    elif train_acc < 0.7:
        status = "📉 Underfitting!"
    else:
        status = "✅ Good fit!"

    print(f"{depth_str:12} {train_acc*100:10.2f}%"
          f" {test_acc*100:8.2f}%  {status}")

# Confusion Matrix
print("\n===== CONFUSION MATRIX (Best Model) =====")
best_model = SVC(kernel="linear")
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred,
                      labels=["Noob","Pro","Legend"])
print("Predicted →  Noob  Pro  Legend")
print(f"Actual Noob  {cm[0]}")
print(f"Actual Pro   {cm[1]}")
print(f"Actual Legend{cm[2]}")

# Metrics Deep Dive
print("\n===== METRICS DEEP DIVE =====")
print(classification_report(y_test, y_pred,
      zero_division=0))

# Visualization
plt.figure(figsize=(10,5))
names = list(cv_results.keys())
means = [cv_results[n].mean()*100 for n in names]
stds = [cv_results[n].std()*100 for n in names]

plt.bar(names, means, yerr=stds,
        capsize=5, color="steelblue",
        error_kw={"ecolor":"red"})
plt.xlabel("Algorithm")
plt.ylabel("CV Accuracy %")
plt.title("5-Fold Cross Validation Results")
plt.xticks(rotation=45)
plt.ylim(0, 120)
for i, (m, s) in enumerate(zip(means, stds)):
    plt.text(i, m+s+2, f"{m:.0f}%",
             ha="center", fontsize=9)
plt.tight_layout()
plt.show()

# Summary
print("\n===== SUMMARY =====")
best_cv = max(cv_results,
              key=lambda x: cv_results[x].mean())
print(f"Best algorithm by CV: {best_cv}")
print(f"CV Accuracy: "
      f"{cv_results[best_cv].mean()*100:.2f}%")
print(f"Std Deviation: "
      f"{cv_results[best_cv].std()*100:.2f}%")
print("\nKey takeaways:")
print("1. CV gives more reliable accuracy than simple split")
print("2. Low std deviation = consistent model")
print("3. Watch for overfitting (train acc >> test acc)")
print("4. Always use CV for final model selection!")