# DAY 23: Decision Trees
# Gaming Rank Prediction

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
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
    "Rank": ["Noob","Noob","Noob","Pro",
             "Pro","Legend","Pro","Legend",
             "Noob","Noob","Noob","Noob",
             "Pro","Legend","Pro","Legend"]
}

df = pd.DataFrame(data)
print(df)

# Features and Labels
X = df[["Level","Wins","Loses"]]
y = df["Rank"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy*100:.2f}%")

# Classification Report
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
print(f"\nFeatures Importance:")
for name, importance in zip(X.columns,
                            model.feature_importances_):
    print(f"{name}: {importance:.2f}")

# Tree Rules
tree_rules = export_text(model,
             feature_names=list(X.columns))
print("\nTree Rules:")
print(tree_rules)

# Predict new players
new_player = pd.DataFrame({
    "Level": [90],
    "Wins": [350],
    "Loses": [100]
})
prediction = model.predict(new_player)
print(f"Player with Level 90, 350 Wins, 100 Loses:")
print(f"Rank: {prediction[0]}")

new_player2 = pd.DataFrame({
    "Level": [10],
    "Wins": [50],
    "Loses": [400]
})
prediction2 = model.predict(new_player2)
print(f"Player with Level 10, 50 Wins, 400 Loses:")
print(f"Rank: {prediction2[0]}")