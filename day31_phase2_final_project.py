# DAY 31: Phase 2 Final Project
# Complete AI Student Performance Analyzer 🎓

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans

print("=" * 50)
print("  AI STUDENT PERFORMANCE ANALYZER 🎓")
print("=" * 50)

# ===== PART 1: DATASET =====
print("\n📊 PART 1: CREATING DATASET")
np.random.seed(42)
n = 200

names = ["Kunal","Sonu","Glou","Vedu","Tantan",
         "Durgesh","Om","Sam","Raj","Priya",
         "Neha","Abhinav","Rohan","Vikram","Arjun",
         "Karan","Riya","Pooja","Amit","Rahul"]

data = {
    "Name": [names[i % len(names)] +
             str(i//len(names)+1)
             if i >= len(names) else names[i]
             for i in range(n)],
    "Age": np.random.randint(17, 23, n),
    "Study_Hours": np.random.uniform(1, 10, n),
    "Attendance": np.random.uniform(50, 100, n),
    "Maths": np.random.randint(30, 100, n),
    "Science": np.random.randint(30, 100, n),
    "English": np.random.randint(30, 100, n),
    "Programming": np.random.randint(30, 100, n),
    "Projects_Done": np.random.randint(0, 10, n),
    "Certifications": np.random.randint(0, 5, n),
    "City": np.random.choice(
        ["Nagpur","Mumbai","Delhi",
         "Pune","Hyderabad"], n),
    "Stream": np.random.choice(
        ["CS","IT","Electronics","Mechanical"], n)
}

df = pd.DataFrame(data)

# Derived features
df["Total_Marks"] = (df["Maths"] + df["Science"] +
                     df["English"] + df["Programming"])
df["Percentage"] = (df["Total_Marks"]/400*100).round(2)
df["Study_Score"] = (
    df["Study_Hours"] * 5 +
    df["Attendance"] * 0.3 +
    df["Projects_Done"] * 3 +
    df["Certifications"] * 5
).round(2)

def get_grade(pct):
    if pct >= 85: return "A+"
    elif pct >= 75: return "A"
    elif pct >= 65: return "B"
    elif pct >= 55: return "C"
    elif pct >= 40: return "D"
    else: return "F"

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

df["Grade"] = df["Percentage"].apply(get_grade)
df["Performance"] = df.apply(get_performance, axis=1)

print(f"Dataset created: {df.shape}")
print(df[["Name","Percentage","Grade",
          "Performance","City"]].head(10))

# ===== PART 2: EDA =====
print("\n📈 PART 2: EXPLORATORY DATA ANALYSIS")
print(f"\nTotal Students: {len(df)}")
print(f"\nPerformance Distribution:")
print(df["Performance"].value_counts())
print(f"\nGrade Distribution:")
print(df["Grade"].value_counts())
print(f"\nAverage Percentage by Stream:")
print(df.groupby("Stream")["Percentage"].mean().round(2))
print(f"\nAverage Percentage by City:")
print(df.groupby("City")["Percentage"].mean().round(2).sort_values(ascending=False))
print(f"\nTop 5 Students:")
print(df.nlargest(5,"Percentage")[
    ["Name","Percentage","Grade","Performance"]])
print(f"\nBottom 5 Students:")
print(df.nsmallest(5,"Percentage")[
    ["Name","Percentage","Grade","Performance"]])

# ===== PART 3: VISUALIZATION =====
print("\n📊 PART 3: VISUALIZATIONS")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

perf_counts = df["Performance"].value_counts()
axes[0,0].bar(perf_counts.index,
              perf_counts.values,
              color=["gold","green","orange","red"])
axes[0,0].set_title("Performance Distribution")
axes[0,0].set_ylabel("Count")

axes[0,1].scatter(df["Study_Hours"],
                  df["Percentage"],
                  c=df["Percentage"],
                  cmap="viridis", alpha=0.6)
axes[0,1].set_title("Study Hours vs Percentage")
axes[0,1].set_xlabel("Study Hours")
axes[0,1].set_ylabel("Percentage")

grade_counts = df["Grade"].value_counts()
axes[0,2].pie(grade_counts.values,
              labels=grade_counts.index,
              autopct="%1.1f%%")
axes[0,2].set_title("Grade Distribution")

city_avg = df.groupby("City")["Percentage"].mean()
axes[1,0].bar(city_avg.index, city_avg.values,
              color="steelblue")
axes[1,0].set_title("City wise Average %")
axes[1,0].tick_params(axis="x", rotation=45)

axes[1,1].scatter(df["Attendance"],
                  df["Percentage"],
                  alpha=0.5, color="purple")
axes[1,1].set_title("Attendance vs Percentage")
axes[1,1].set_xlabel("Attendance %")
axes[1,1].set_ylabel("Percentage %")

stream_avg = df.groupby("Stream")["Percentage"].mean()
axes[1,2].bar(stream_avg.index, stream_avg.values,
              color="coral")
axes[1,2].set_title("Stream wise Average %")
axes[1,2].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

# ===== PART 4: ML CLASSIFICATION =====
print("\n🤖 PART 4: ML CLASSIFICATION")
features = ["Age","Study_Hours","Attendance",
            "Maths","Science","English",
            "Programming","Projects_Done",
            "Certifications","Study_Score"]

X = df[features]
y = df["Performance"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

algorithms = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(
                           max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(
                     max_depth=5),
    "Random Forest": RandomForestClassifier(
                     n_estimators=100),
    "SVM": SVC(kernel="rbf")
}

print(f"\n{'Algorithm':20} {'Accuracy':10} {'CV Mean':10}")
print("-" * 42)

results = {}
for name, algo in algorithms.items():
    algo.fit(X_train_scaled, y_train)
    pred = algo.predict(X_test_scaled)
    acc = accuracy_score(y_test, pred)

    # Use cv=2 to avoid warning with small classes
    cv_scores = cross_val_score(
        algo, X_train_scaled, y_train, cv=2)
    cv_mean = cv_scores.mean()

    results[name] = {"acc": acc, "cv": cv_mean}
    print(f"{name:20} {acc*100:8.2f}%  "
          f"{cv_mean*100:8.2f}%")

best_name = max(results,
                key=lambda x: results[x]["acc"])
print(f"\n🏆 Best Algorithm: {best_name}")
print(f"🎯 Accuracy: {results[best_name]['acc']*100:.2f}%")

best_model = algorithms[best_name]
y_pred = best_model.predict(X_test_scaled)
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=le.classes_,
      zero_division=0))

if best_name == "Random Forest":
    print("\nFeature Importance:")
    for feat, imp in sorted(
        zip(features, best_model.feature_importances_),
        key=lambda x: x[1], reverse=True):
        bar = "█" * int(imp * 50)
        print(f"{feat:15}: {bar} {imp:.4f}")

# ===== PART 5: CLUSTERING =====
print("\n🔍 PART 5: STUDENT CLUSTERING")
X_cluster = df[["Study_Hours","Percentage",
                "Attendance","Study_Score"]]
X_cluster_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=4,
                random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_cluster_scaled)

cluster_means = df.groupby("Cluster")["Percentage"].mean()
sorted_clusters = cluster_means.sort_values()
cluster_names = {}
cluster_labels = ["Struggling 😔","Average 😐",
                  "Good 😊","Excellent 🌟"]
for i, (cluster, _) in enumerate(
                        sorted_clusters.items()):
    cluster_names[cluster] = cluster_labels[i]

df["Cluster_Name"] = df["Cluster"].map(cluster_names)

print("\nCluster Distribution:")
print(df["Cluster_Name"].value_counts())
print("\nCluster Analysis:")
for cluster_name in cluster_labels:
    cluster_data = df[df["Cluster_Name"]==cluster_name]
    if len(cluster_data) > 0:
        print(f"\n{cluster_name}:")
        print(f"  Students: {len(cluster_data)}")
        print(f"  Avg %: {cluster_data['Percentage'].mean():.1f}%")
        print(f"  Avg Study: {cluster_data['Study_Hours'].mean():.1f} hrs")

# ===== PART 6: PREDICT NEW STUDENTS =====
print("\n🔮 PART 6: PREDICT NEW STUDENTS")
new_students = pd.DataFrame({
    "Name": ["Kunal C","New Student","Average Joe"],
    "Age": [20, 19, 21],
    "Study_Hours": [8, 2, 5],
    "Attendance": [90, 60, 75],
    "Maths": [90, 45, 65],
    "Science": [85, 40, 60],
    "English": [88, 50, 62],
    "Programming": [95, 35, 58],
    "Projects_Done": [5, 1, 3],
    "Certifications": [2, 0, 1],
    "Study_Score": [0, 0, 0]
})

new_students["Study_Score"] = (
    new_students["Study_Hours"] * 5 +
    new_students["Attendance"] * 0.3 +
    new_students["Projects_Done"] * 3 +
    new_students["Certifications"] * 5
).round(2)

X_new = new_students[features]
X_new_scaled = scaler.transform(X_new)
predictions = best_model.predict(X_new_scaled)
predicted_labels = le.inverse_transform(predictions)

print(f"\n{'Student':15} {'Predicted Performance':20}")
print("-" * 37)
for name, pred in zip(new_students["Name"],
                       predicted_labels):
    emoji = "🌟" if pred=="Excellent" else \
            "😊" if pred=="Good" else \
            "😐" if pred=="Average" else "😔"
    print(f"{name:15} {pred} {emoji}")

# ===== PART 7: FINAL SUMMARY =====
print("\n" + "=" * 50)
print("  FINAL PROJECT SUMMARY")
print("=" * 50)
print(f"✅ Dataset: {len(df)} students analyzed")
print(f"✅ Features: {len(features)} input features")
print(f"✅ Algorithms tested: {len(algorithms)}")
print(f"✅ Best Algorithm: {best_name}")
print(f"✅ Best Accuracy: {results[best_name]['acc']*100:.2f}%")
print(f"✅ Clusters found: 4 student groups")
print(f"✅ New students predicted: {len(new_students)}")
print(f"\n📊 Performance Distribution:")
for perf, count in df["Performance"].value_counts().items():
    pct = count/len(df)*100
    bar = "█" * int(pct/5)
    print(f"  {perf:10}: {bar} {count} ({pct:.1f}%)")
print(f"\n🎓 Grade Distribution:")
for grade, count in df["Grade"].value_counts().items():
    print(f"  Grade {grade}: {count} students")
print("\n🏆 Phase 2 Machine Learning — COMPLETE!")
print("🚀 Ready for Phase 3: Deep Learning!")