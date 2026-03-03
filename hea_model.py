import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Dummy dataset (replace later with real HEA data)
data = {
    "Fe": [20, 25, 30, 35, 40],
    "Ni": [20, 25, 20, 25, 20],
    "Cr": [20, 15, 20, 15, 20],
    "Co": [20, 20, 15, 15, 10],
    "Hardness": [200, 220, 250, 270, 300]
}

df = pd.DataFrame(data)

X = df[["Fe", "Ni", "Cr", "Co"]]
y = df["Hardness"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "hea_model.pkl")

print("Model trained and saved.")
