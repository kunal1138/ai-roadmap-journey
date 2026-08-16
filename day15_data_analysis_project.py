# DAY 15: Data Analysis Mini Project - Student Result System

import numpy as np
import pandas as pd

# Creating student dataset
data = {
    "Name": ["Kunal", "Sonu", "Dimpesh", "Om",
             "Abhinav", "Raj", "Priya", "Neha"],
    "Age": [20, 21, 22, 16, 20, 19, 21, 18],
    "Maths": [95, 80, 45, 60, 88, 72, 55, 90],
    "Science": [90, 75, 50, 65, 92, 68, 48, 85],
    "English": [85, 70, 55, 70, 78, 65, 60, 88],
    "City": ["Nagpur", "Delhi", "Mumbai",
             "Nagpur", "Nagpur", "Pune",
             "Delhi", "Nagpur"]
}

# Creating DataFrame
df = pd.DataFrame(data)
print(df)

# Calculate Total and Percentage
df["Total"] = df["Maths"] + df["Science"] + df["English"]
df["Percentage"] = (df["Total"] / 300) * 100

# Grade function
def get_grade(percentage):
    if percentage >= 90:
        return "A+"
    elif percentage >= 80:
        return "A"
    elif percentage >= 70:
        return "B"
    elif percentage >= 60:
        return "C"
    elif percentage >= 40:
        return "D"
    else:
        return "F"

# Apply grade function
df["Grade"] = df["Percentage"].apply(get_grade)

# Pass/Fail using lambda
df["Result"] = df["Percentage"].apply(
    lambda x: "Pass" if x >= 40 else "Fail"
)

# Print full result
print("===== STUDENT RESULT ANALYSIS =====")
print(df)

# Class Statistics
print("===== CLASS STATICS =====")
print(f"Highest percentage:{df['Percentage'].max():.2f}%")
print(f"Lowest percentage:{df['Percentage'].min():.2f}%")
print(f"Class average:{df['Percentage'].mean():.2f}%")
print(f"Total Students:{len(df)}")
print(f"Passed:{len(df[df['Result']=='Pass'])}")
print(f"Failed:{len(df[df['Result']=='Fail'])}")

# Topper
topper = df[df["Percentage"] == df["Percentage"].max()]
print(f"\n🏆 Topper:{topper['Name'].values[0]} with {topper['Percentage'].values[0]:.2f}%")

# City wise average
print("===== CITY WISE AVERAGE =====")
print(df.groupby("City")["Percentage"].mean())

# Nagpur students
print("===== NAGPUR STUDENTS =====")
print(df[df["City"] == "Nagpur"][["Name", "Percentage", "Grade"]])

# Rank list
print("===== RANK LIST =====")
df["Rank"] = df["Percentage"].rank(ascending=False).astype(int)
print(df[["Rank", "Name", "Percentage", "Grade", "Result"]].sort_values("Rank"))