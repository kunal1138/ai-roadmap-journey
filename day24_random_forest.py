# DAY 24: Random Forest
# Gaming Rank Prediction with 100 Trees

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

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

# Random Forest with 100 trees
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy*100:.2f}%")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      zero_division=0))

# Feature Importance
print("\nFeatures Importance:")
for name, importance in zip(X.columns,
                            model.feature_importances_):
    print(f"{name}: {importance:.4f}")

# Decision Tree for comparison
dt_model = DecisionTreeClassifier(max_depth=3,
                                   random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)

# Comparison
print("\n==== Comparison ====")
print(f"Decision Tree Accuracy: {dt_accuracy*100:.2f}%")
print(f"Random Forest Accuracy: {accuracy*100:.2f}%")
print(f"Winner: {'Random Forest 🌲' if accuracy > dt_accuracy else 'Decision Tree 🌳'}")

# Predict new players
new_player = pd.DataFrame({
    "Level": [94],
    "Wins": [400],
    "Loses": [50],
    "Hours": [10]
})
prediction = model.predict(new_player)
print(f"\nPlayer: Level 94, 400 Wins, 50 Loses, 10 Hours")
print(f"Rank: {prediction[0]}")

new_player2 = pd.DataFrame({
    "Level": [5],
    "Wins": [20],
    "Loses": [200],
    "Hours": [1]
})
prediction2 = model.predict(new_player2)
print(f"\nPlayer: Level 5, 20 Wins, 200 Loses, 1 Hour")
print(f"Rank: {prediction2[0]}")