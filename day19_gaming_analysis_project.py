# DAY 19: Week 3 Mini Project - Gaming Analysis System 🎮

import numpy as np
import pandas as pd

# Gaming dataset - Friend group analysis
data = {
    "Persons": ["Kunal","Sonu","Glou","Vedu","Tantan","Durgesh","Om","Sam"],
    "Games": ["PUBG","FREE FIRE","COD","Fortnite","Clash Royal","COC","Lords Mobile","Asphalt Unit"],
    "Type": ["action","Role Playing","Strategic","Adventure","Combat","Multiplayer","Competative","Timable"],
    "Played": [454,345,654,345,644,445,445,555],
    "Wins": [232,234,223,123,342,345,234,345],
    "Loses": [123,454,453,345,345,344,456,432],
    "Level": [12,23,24,67,78,87,88,99],
    "Role": ["Assualter","Defend","FREE Style","Gamer","Camper","All Rounder","Looser","Winner"]
}

# Creating DataFrame
df = pd.DataFrame(data)

# Calculate Win/Loss Average
df["Avg"] = (df["Wins"]/df["Loses"]).round(2)

# Calculate Score
df["Score"] = (
    (df["Played"]*0.5) +
    (df["Wins"]*20) +
    (df["Loses"]*5)
)

print(df["Score"])

# Rating function
def get_rating(Score):
    if Score >= 9000:
        return "Universal MVP"
    elif Score >= 8000:
        return "MVP"
    elif Score >= 7000:
        return "Legend"
    elif Score >= 5000:
        return "Pro Players"
    else:
        return "Noob"

# Apply rating
df["Rating"] = df["Score"].apply(get_rating)

# Game Analysis
print("==== Game Analysis ====")
print(df[["Persons","Games","Type","Played","Wins","Loses","Level","Role"]])

# Top Scorer
print("\n==== Top Scorer ====")
top = df[df["Wins"] == df["Wins"].max()]
print(f"{top['Persons'].values[0]} with {top['Wins'].values[0]} yeah.")

# Top Loser
print("\n==== Top Loser ====")
back = df[df["Loses"] == df["Loses"].max()]
print(f"{back['Persons'].values[0]} with {back['Loses'].values[0]} shit..!")

# Average wins per person
print("\n==== Average Person ====")
print(df.groupby("Persons")["Wins"].mean().round(2))

# Win/Loss ratio analysis
print("\n==== Played wise Analysis ====")
print(df.groupby("Persons")["Avg"].mean().round(2))

# Statistics
print("\n==== STATICS ====")
print(f"Total played matches : {df['Played'].sum()}")
print(f"Total Wins : {df['Wins'].sum()}")
print(f"Total Loses : {df['Loses'].sum()}")
print(f"Average Wins Player : {df['Wins'].mean():.2f}")
print(f"Std Deviation of Matches : {np.std(df['Played']):.2f}")

# Player Rankings
df["Rank"] = df["Score"].rank(ascending=False).astype(int)
print("\n==== RANKINGS ====")
print(df[["Rank","Persons","Role","Score","Rating"]].sort_values("Rank"))