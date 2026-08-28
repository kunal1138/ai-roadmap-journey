# DAY 27: ML Mini Project
# Complete Player Rank Prediction System 🎮

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans

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

# Dataset Analysis
print("===== PLAYER DATASET =====")
print(df[["Player","Level","Wins","Loses",
          "Hours","Rank"]])
print("\n===== DATASET ANALYSIS =====")
print(f"Total Players: {len(df)}")
print(f"Noobs: {len(df[df['Rank']=='Noob'])}")
print(f"Pros: {len(df[df['Rank']=='Pro'])}")
print(f"Legends: {len(df[df['Rank']=='Legend'])}")
print(f"\nAverage Level: {df['Level'].mean():.1f}")
print(f"Average Wins: {df['Wins'].mean():.1f}")
print(f"Average Loses: {df['Loses'].mean():.1f}")

# Features and Labels
X = df[["Level","Wins","Loses","Hours"]]
y = df["Rank"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# All algorithms
print("\n===== ALGORITHM COMPARISON =====")
algorithms = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(max_depth=3),
    "Random Forest": RandomForestClassifier(
                     n_estimators=100),
    "SVM Linear": SVC(kernel="linear"),
    "SVM RBF": SVC(kernel="rbf")
}

results = {}
for name, algo in algorithms.items():
    algo.fit(X_train_scaled, y_train)
    pred = algo.predict(X_test_scaled)
    acc = accuracy_score(y_test, pred)
    results[name] = acc
    print(f"{name:20}: {acc*100:.2f}%")

# Best Algorithm
best_algo_name = max(results, key=results.get)
best_accuracy = results[best_algo_name]
print(f"\n🏆 Best Algorithm: {best_algo_name}")
print(f"🎯 Best Accuracy: {best_accuracy*100:.2f}%")

# Best Model Details
print(f"\n===== {best_algo_name} DETAILS =====")
best_model = algorithms[best_algo_name]
y_pred = best_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred,
      zero_division=0))

# Bar Chart
plt.figure(figsize=(10,5))
names = list(results.keys())
accuracies = [v*100 for v in results.values()]
colors = ["green" if v==max(accuracies)
          else "blue" for v in accuracies]
plt.bar(names, accuracies, color=colors)
plt.xlabel("Algorithm")
plt.ylabel("Accuracy %")
plt.title("Algorithm Comparison - Player Rank")
plt.xticks(rotation=45)
plt.ylim(0, 110)
for i, v in enumerate(accuracies):
    plt.text(i, v+1, f"{v:.0f}%", ha="center")
plt.tight_layout()
plt.show()

# K-Means Clustering
print("\n===== UNSUPERVISED CLUSTERING =====")
X_all = df[["Level","Wins","Loses","Hours"]]
X_all_scaled = scaler.fit_transform(X_all)
kmeans = KMeans(n_clusters=3,
                random_state=42,
                n_init=10)
df["Cluster"] = kmeans.fit_predict(X_all_scaled)

cluster_means = df.groupby("Cluster")["Level"].mean()
sorted_clusters = cluster_means.sort_values()
cluster_names = {}
labels = ["Noob 🔰", "Pro ⚡", "Legend 🏆"]
for i, (cluster, _) in enumerate(
                        sorted_clusters.items()):
    cluster_names[cluster] = labels[i]

df["Cluster_Name"] = df["Cluster"].map(cluster_names)
print(df[["Player","Rank","Cluster_Name"]])

# Predict New Players
print("\n===== PREDICT NEW PLAYERS =====")
new_players = pd.DataFrame({
    "Player": ["Champion","Beginner","Fighter"],
    "Level": [95, 10, 60],
    "Wins": [400, 30, 250],
    "Loses": [50, 500, 200],
    "Hours": [10, 1, 6]
})

print("New Players:")
print(new_players)

X_new = new_players[["Level","Wins","Loses","Hours"]]
X_new_scaled = scaler.transform(X_new)

print(f"\nPredictions using {best_algo_name}:")
predictions = best_model.predict(X_new_scaled)
for player, pred in zip(new_players["Player"],
                         predictions):
    emoji = "🏆" if pred=="Legend" else \
            "⚡" if pred=="Pro" else "🔰"
    print(f"{player:10} → {pred} {emoji}")

# Final Summary
print("\n===== PROJECT SUMMARY =====")
print(f"Total Players Analyzed: {len(df)}")
print(f"Best Algorithm: {best_algo_name}")
print(f"Best Accuracy: {best_accuracy*100:.2f}%")
print(f"New Players Predicted: {len(new_players)}")
print("\nAlgorithms tested:")
for name, acc in results.items():
    star = "⭐" if name==best_algo_name else "  "
    print(f"{star} {name:20}: {acc*100:.2f}%")