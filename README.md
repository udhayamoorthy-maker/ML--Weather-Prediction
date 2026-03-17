# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset, remove unnecessary columns, handle missing values, and encode categorical features.

2.Split the data into input features (X) and target variables (Temperature, PM2.5, Energy).

3.Divide the dataset into training and testing sets, then train a Random Forest Regressor model.

4.Use the trained model to predict values on the test set and display the results.

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: udhayamoorthy A
RegisterNumber:  212225040477
*/
```
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("Data.csv")

# Drop unused columns
df = df.drop(columns=["time", "bat"], errors='ignore')

# Handle missing values
df = df.dropna()

# Encode categorical column
if "wind_direction" in df.columns:
    le = LabelEncoder()
    df["wind_direction"] = le.fit_transform(df["wind_direction"])

# Features and targets
X = df.drop(columns=["tem", "pm2_5", "tsr"])
y = df[["tem", "pm2_5", "tsr"]]

# Split (with reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model (with reproducibility)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Display results
output = pd.DataFrame(predictions, columns=["Temperature", "PM2.5", "Energy"])
print(output)
```

## Output:

<img width="588" height="319" alt="564641261-7e84fc1c-a29f-4ee9-833f-6970b6723b1c" src="https://github.com/user-attachments/assets/72a46a29-782e-474d-bf11-d16d77dfa056" />


## Result:
The program was successfully executed and verified successfully.
