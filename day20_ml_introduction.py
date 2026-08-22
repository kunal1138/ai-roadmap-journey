# DAY 20: Introduction to Machine Learning
# First ML Model - KNN Classifier on Iris Dataset

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()

# Features (Input - X)
X = iris.data
print("Features Shape:", X.shape)
print("Features names:", iris.feature_names)

# Labels (Output - y)
y = iris.target
print("Labels:", iris.target_names)

# Split data - 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training Sample: {len(X_train)}")
print(f"Testing Sample: {len(X_test)}")

# Create and train KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy*100:.2f}%")

# Predict new flowers
new_flower = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(new_flower)
print(f"New flower is: {iris.target_names[prediction[0]]}")