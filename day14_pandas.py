# DAY 14: Pandas - DataFrames and Operations

import numpy as np
import pandas as pd

# Creating a Series
s = pd.Series([1,2,3,4,5])
print(s)

# Creating a Dictionary
data = {
    "Name": ["kunal", "sonu", "dimpesh", "om"],
    "age": [20, 21, 22, 16],
    "marks": [100, 90, 70, 50],
    "place": ["nagpur", "delhi", "mumbai", "mumbai"]
}

# Creating DataFrame
df = pd.DataFrame(data)
print(df)

# Basic Operations
print(df.head())        # first 5 rows
print(df.tail())        # last 5 rows
print(df.shape)         # (4, 4) - 4 rows, 4 columns
print(df.info())        # data types and memory
print(df.describe())    # statistics - mean, std, min, max

# Selecting single column
print(df["Name"])

# Selecting multiple columns
print(df[["Name", "marks"]])

# Filtering - marks greater than 50
print(df[df["marks"] > 50])

# Filtering - marks less than 50
print(df[df["marks"]