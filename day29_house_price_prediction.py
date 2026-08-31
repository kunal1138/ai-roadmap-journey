# DAY 29: House Price Prediction
# Nagpur Real Estate ML Project 🏠

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Create Nagpur house dataset
np.random.seed(42)
n = 100

data = {
    "Area_sqft": np.random.randint(500, 3000, n),
    "Bedrooms": np.random.randint(1, 5, n),
    "Bathrooms": np.random.randint(1, 4, n),
    "Floor": np.random.randint(0, 15, n),
    "Age_years": np.random.randint(0, 30, n),
    "Distance_center_km": np.random.uniform(1, 30, n),
    "Parking": np.random.randint(0, 2, n),
    "Locality": np.random.choice(
        ["Dharampeth","Sitabuldi","Wardha Road",
         "Hingna","Kamptee"], n)
}

df = pd.DataFrame(data)

# Create realistic prices
df["Price_lakhs"] = (
    (df["Area_sqft"] * 0.08) +
    (df["Bedrooms"] * 5) +
    (df["Bathrooms"] * 3) +
    (df["Floor"] * 1.5) -
    (df["Age_years"] * 0.5) -
    (df["Distance_center_km"] * 1.2) +
    (df["Parking"] * 3) +
    np.random.normal(0, 5, n)
).round(2)

# Dataset Overview
print("===== NAGPUR HOUSE DATASET =====")
print(df.head(10))
print(f"\nDataset Shape: {df.shape}")

# EDA
print("\n===== EXPLORATORY DATA ANALYSIS =====")
print(df.describe())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== PRICE STATISTICS =====")
print(f"Min Price: ₹{df['Price_lakhs'].min():.2f} Lakhs")
print(f"Max Price: ₹{df['Price_lakhs'].max():.2f} Lakhs")
print(f"Avg Price: ₹{df['Price_lakhs'].mean():.2f} Lakhs")
print(f"Median Price: ₹{df['Price_lakhs'].median():.2f} Lakhs")

# Locality Analysis
print("\n===== LOCALITY WISE PRICES =====")
locality_avg = df.groupby("Locality")["Price_lakhs"].mean().round(2)
print(locality_avg.sort_values(ascending=False))

# Correlation
print("\n===== FEATURE CORRELATION =====")
numeric_df = df.select_dtypes(include=[np.number])
correlation = numeric_df.corr()["Price_lakhs"].sort_values(ascending=False)
print(correlation)

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0,0].hist(df["Price_lakhs"], bins=20,
               color="steelblue", edgecolor="white")
axes[0,0].set_title("Price Distribution")
axes[0,0].set_xlabel("Price (Lakhs)")

axes[0,1].scatter(df["Area_sqft"],
                  df["Price_lakhs"],
                  alpha=0.6, color="green")
axes[0,1].set_title("Area vs Price")
axes[0,1].set_xlabel("Area (sqft)")
axes[0,1].set_ylabel("Price (Lakhs)")

bedroom_avg = df.groupby("Bedrooms")["Price_lakhs"].mean()
axes[1,0].bar(bedroom_avg.index,
              bedroom_avg.values,
              color="orange")
axes[1,0].set_title("Bedrooms vs Avg Price")
axes[1,0].set_xlabel("Bedrooms")
axes[1,0].set_ylabel("Avg Price (Lakhs)")

locality_avg.sort_values().plot(kind="barh",
                                ax=axes[1,1],
                                color="purple")
axes[1,1].set_title("Locality vs Avg Price")
axes[1,1].set_xlabel("Avg Price (Lakhs)")

plt.tight_layout()
plt.show()

# Feature Engineering
print("\n===== FEATURE ENGINEERING =====")
df["Locality_encoded"] = pd.factorize(df["Locality"])[0]
df["Price_per_sqft"] = (df["Price_lakhs"] /
                         df["Area_sqft"] * 100).round(2)

print("New features added:")
print(df[["Locality","Locality_encoded",
          "Price_per_sqft"]].head())

# Model Training
print("\n===== MODEL TRAINING =====")
features = ["Area_sqft","Bedrooms","Bathrooms",
            "Floor","Age_years",
            "Distance_center_km",
            "Parking","Locality_encoded"]

X = df[features]
y = df["Price_lakhs"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
print("\n--- Linear Regression ---")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

lr_r2 = r2_score(y_test, lr_pred)
lr_mse = mean_squared_error(y_test, lr_pred)
lr_rmse = np.sqrt(lr_mse)

print(f"R² Score: {lr_r2:.4f}")
print(f"MSE: {lr_mse:.2f}")
print(f"RMSE: {lr_rmse:.2f} Lakhs")

print("\nFeature Coefficients:")
for feat, coef in zip(features, lr.coef_):
    print(f"{feat:25}: {coef:.4f}")

# Random Forest
print("\n--- Random Forest Regressor ---")
rf = RandomForestRegressor(n_estimators=100,
                            random_state=42)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)

rf_r2 = r2_score(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)

print(f"R² Score: {rf_r2:.4f}")
print(f"MSE: {rf_mse:.2f}")
print(f"RMSE: {rf_rmse:.2f} Lakhs")

print("\nFeature Importance:")
for feat, imp in sorted(
    zip(features, rf.feature_importances_),
    key=lambda x: x[1], reverse=True):
    print(f"{feat:25}: {imp:.4f}")

# Model Comparison
print("\n===== MODEL COMPARISON =====")
print(f"{'Model':20} {'R²':10} {'RMSE':10}")
print("-" * 42)
print(f"{'Linear Regression':20} "
      f"{lr_r2:10.4f} {lr_rmse:8.2f} Lakhs")
print(f"{'Random Forest':20} "
      f"{rf_r2:10.4f} {rf_rmse:8.2f} Lakhs")

winner = "Linear Regression" if lr_r2 > rf_r2 \
         else "Random Forest"
print(f"\n🏆 Best Model: {winner}")

# Predict New Houses
print("\n===== PREDICT HOUSE PRICES =====")
houses = pd.DataFrame({
    "Area_sqft": [1200, 2500, 800],
    "Bedrooms": [2, 4, 1],
    "Bathrooms": [2, 3, 1],
    "Floor": [3, 8, 1],
    "Age_years": [5, 2, 15],
    "Distance_center_km": [5, 15, 25],
    "Parking": [1, 1, 0],
    "Locality_encoded": [0, 1, 2]
})

houses_scaled = scaler.transform(houses)
best_model = lr if lr_r2 > rf_r2 else rf
predictions = best_model.predict(houses_scaled)

print(f"{'House':8} {'Area':8} {'BHK':6} "
      f"{'Predicted Price':15}")
print("-" * 40)
for i, (_, house) in enumerate(houses.iterrows()):
    bhk = f"{int(house['Bedrooms'])}BHK"
    area = f"{int(house['Area_sqft'])}sqft"
    price = f"₹{predictions[i]:.2f} Lakhs"
    print(f"House {i+1}  {area:8} {bhk:6} {price}")

# Summary
print("\n===== PROJECT SUMMARY =====")
print(f"Dataset: 100 Nagpur houses")
print(f"Features used: {len(features)}")
print(f"Best Model: {winner}")
print(f"Best R² Score: {max(lr_r2, rf_r2):.4f}")
print(f"Best RMSE: {min(lr_rmse, rf_rmse):.2f} Lakhs")