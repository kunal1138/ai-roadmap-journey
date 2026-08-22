# DAY 21: Linear Regression
# Study Hours vs Marks Prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Dataset - Study hours vs Marks
data = {
    "Study_Hours": [1,2,3,4,5,6,7,8,9,10],
    "Marks": [35,45,50,60,65,70,75,85,90,95]
}

df = pd.DataFrame(data)
print(df)

# Features and Labels
X = df[["Study_Hours"]]  # input
y = df["Marks"]           # output

# Split data - 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Model details
print(f"Slope: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Predict new value
hours = np.array([[7.5]])
predicted_marks = model.predict(hours)
print(f"\nIf you study 7.5 hours → Marks: {predicted_marks[0]:.2f}")

# Plot the graph
plt.scatter(X, y, color="blue", label="Actual")
plt.plot(X, model.predict(X), color="red", label="Predicted Line")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks - Linear Regression")
plt.legend()
plt.show()