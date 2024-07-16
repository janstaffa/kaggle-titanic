import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("data/train.csv")
print(df.columns)


# Engineer new features
decks = ["A", "B", "C", "D", "E", "F", "G"]
def extract_deck_from_cabin(cabin):
    if pd.isna(cabin):
        return np.nan

    return cabin[0] if cabin[0] in decks else np.nan

df["Deck"] = df["Cabin"].map(extract_deck_from_cabin)
df["FamilySize"] = df["SibSp"] + df["Parch"]


survivors = df[df["Survived"] == 1]

print(str(round(len(survivors) / len(df) * 100, 1)) + "% survived")

# Class
first_class_passengers = df[df['Pclass'] == 1]
first_class_survivors = survivors[survivors['Pclass'] == 1]

print("\n== Class ==")
print(str(round(len(first_class_survivors)/len(first_class_passengers) * 100, 1)) + "% of first class passengers survived")
print(str(round(len(first_class_survivors)/len(survivors) * 100, 1)) + "% of survivors were first class")


# Gender
female_passengers = df[df['Sex'] == "female"]
female_survivors = survivors[survivors['Sex'] == "female"]
male_passengers = df[df['Sex'] == "male"]
male_survivors = survivors[survivors['Sex'] == "male"]

print("\n== Gender ==")
print(str(round(len(female_survivors)/len(female_passengers) * 100, 1)) + "% of female passengers survived")
print(str(round(len(female_survivors)/len(survivors) * 100, 1)) + "% of survivors were female")

print(str(round(len(male_survivors)/len(male_passengers) * 100, 1)) + "% of male passengers survived")
print(str(round(len(male_survivors)/len(survivors) * 100, 1)) + "% of survivors were male")


# Age
survivor_ages = survivors['Age'].dropna()

survivor_age_mean = survivor_ages.mean()
survivor_age_median = survivor_ages.median()
survivor_age_mode = stats.mode(survivor_ages)[0]



print("\n== Age ==")
print("Survivor age MEAN: " + str(round(survivor_age_mean, 1)))
print("Survivor age MEDIAN: " + str(round(survivor_age_median, 1)))
print("Survivor age MODE: " + str(round(survivor_age_mode, 1)))


age_below_mean =  df[df['Age'] <= survivor_age_mean]
age_below_median = df[df['Age'] <= survivor_age_median]
age_below_mode = df[df['Age'] <= survivor_age_mode]

survivor_age_below_mean = survivors[survivors['Age'] <= survivor_age_mean]
survivor_age_below_median = survivors[survivors['Age'] <= survivor_age_median]
survivor_age_below_mode = survivors[survivors['Age'] <= survivor_age_mode]



print(str(round(len(survivor_age_below_mean) / len(age_below_mean) * 100, 1)) + "% of passengers bellow survivor age mean survived")
print(str(round(len(survivor_age_below_median) / len(age_below_median) * 100, 1)) + "% of passengers bellow survivor age median survived")
print(str(round(len(survivor_age_below_mode) / len(age_below_mode) * 100, 1)) + "% of passengers bellow survivor age mode survived")

# Fare
avg_fare = df['Fare'].mean()
above_avg_fare_passengers = df[df['Fare'] >= avg_fare]
above_avg_fare_survivors = survivors[survivors['Fare'] >= avg_fare]

print("\n== Fare ==")
print(str(round(len(above_avg_fare_survivors) / len(above_avg_fare_passengers) * 100, 1)) + "% of passangers that paid above average fare survived")
print(str(round(len(above_avg_fare_survivors) / len(survivors) * 100, 1)) + "% of survivors paid above average fare")


# Family size
avg_family_size = df['FamilySize'].mean()
passengers_above_avg_family_size = df[df["FamilySize"] >= avg_family_size]
passengers_below_avg_family_size = df[df["FamilySize"] < avg_family_size]

survivors_above_avg_family_size = survivors[survivors["FamilySize"] >= avg_family_size]
survivors_below_avg_family_size = survivors[survivors["FamilySize"] < avg_family_size]

print("\n== Family size ==")
print("Average family size: " + str(round(avg_family_size, 1)))
print(str(round(len(survivors_above_avg_family_size) / len(survivors)* 100, 1)) + "% of survivors had an above average family size")
print(str(round(len(survivors_below_avg_family_size) / len(survivors)* 100, 1)) + "% of survivors had a below average family size")
print(str(round(len(survivors_above_avg_family_size) / len(passengers_above_avg_family_size)* 100, 1)) + "% of above average family size passengers survived")
print(str(round(len(survivors_below_avg_family_size) / len(passengers_below_avg_family_size)* 100, 1)) + "% of below average family size passengers survived")

for name, group in df.groupby('FamilySize'):
    g_survivors = group[group['Survived'] == 1]
    print(f"Family size: {name} - {str(round(len(g_survivors) / len(group) * 100, 1))}% survived")

# print(df.tail())