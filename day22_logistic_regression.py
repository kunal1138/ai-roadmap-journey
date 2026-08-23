# DAY 22: Logistic Regression
# Study Hours vs Pass/Fail Prediction

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Dataset
data = {
    "Study_Hours": [1,2,3,4,5,6,7,8,9,10,
                    1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,11],
    "Marks": [30,40,45,55,60,65,75,80,90,95,
              35,42,48,58,62,68,78,82,92,98],
    "Result": [0,0,0,0,1,1,1,1,1,1,
               0,0,0,0,1,1,1,1,1,1]
    # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)
print(df)

# Features and Labels
X = df[["Study_Hours", "Marks"]]
y = df["Result"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Predict new students
new_student = pd.DataFrame({
    "Study_Hours": [3],
    "Marks": [45]
})
prediction = model.predict(new_student)
print(f"\nStudent with 3hours study, 45 Marks")
print(f"Result: {'Pass ✅' if prediction[0]==1 else 'Fail ❌'}")

new_student2 = pd.DataFrame({
    "Study_Hours": [7],
    "Marks": [75]
})
prediction2 = model.predict(new_student2)
print(f"\nStudent with 7hours study, 75 Marks")
print(f"Result: {'Pass ✅' if prediction2[0]==1 else 'Fail ❌'}")