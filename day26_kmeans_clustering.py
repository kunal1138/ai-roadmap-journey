# DAY 26: K-Means Clustering
# Gaming Player Grouping - Unsupervised Learning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Gaming dataset - NO labels!
data = {
    "Level": [12,23,24,67,78,87,88,99,
              15,25,35,45,55,65,75,85],
    "Wins": [232,234,223,123,342,345,234,345,
             150,200,220,180,300,320,280,340],
    "Loses": [123,454,453,345,345,344,456,432,
              200,300,250,400,200,150,300,250],
    "Hours": [2,3,4,5,6,7,8,9,
              1,2,3,4,5,6,7,8]
}

df = pd.DataFrame(data)
print(df)

# Features only - no labels!
X = df[["Level","Wins","Loses","Hours"]]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method
print("\n===== ELBOW METHOD =====")
inertias = []
k_values = range(1, 8)

for k in k_values:
    kmeans = KMeans(n_clusters=k,
                    random_state=42,
                    n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    print(f"K={k}: Inertia={kmeans.inertia_:.2f}")

# Plot Elbow
plt.figure(figsize=(8,4))
plt.plot(list(k_values), inertias, "bo-")
plt.xlabel("Number of Cluster (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method - Finding best k")
plt.xticks(list(k_values))
plt.show()

# K-Means with K=3
print("\n===== K-MEANS WITH K=3 =====")
kmeans = KMeans(n_clusters=3,
                random_state=42,
                n_init=10)
kmeans.fit(X_scaled)

# Add clusters to dataframe
df["Cluster"] = kmeans.labels_
print(df)

# Cluster Analysis
print("\n===== CLUSTER ANALYSIS =====")
for i in range(3):
    cluster_data = df[df["Cluster"]==i]
    print(f"\nCluster {i}:")
    print(f"Size: {len(cluster_data)} players")
    print(f"Avg Level: {cluster_data['Level'].mean():.1f}")
    print(f"Avg Wins: {cluster_data['Wins'].mean():.1f}")
    print(f"Avg Loses: {cluster_data['Loses'].mean():.1f}")
    print(f"Avg Hours: {cluster_data['Hours'].mean():.1f}")

# Name clusters
print("\n===== CLUSTER NAMES =====")
cluster_means = df.groupby("Cluster")["Level"].mean()
sorted_clusters = cluster_means.sort_values()

for rank, (cluster, mean_level) in enumerate(
                                sorted_clusters.items()):
    if rank == 0:
        name = "Noob Group 🔰"
    elif rank == 1:
        name = "Pro Group ⚡"
    else:
        name = "Legend Group 🏆"
    print(f"Cluster {cluster} → {name} "
          f"(Avg Level: {mean_level:.1f})")

# Predict new player
new_player = pd.DataFrame({
    "Level": [90],
    "Wins": [380],
    "Loses": [80],
    "Hours": [9]
})
new_scaled = scaler.transform(new_player)
cluster = kmeans.predict(new_scaled)
print(f"\nNew Player (Level 90, 380 Wins):")
print(f"Cluster: {cluster[0]} → Legend Group 🏆")