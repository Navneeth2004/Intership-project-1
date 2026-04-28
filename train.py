import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

os.makedirs("model", exist_ok=True)

# Load data
df = pd.read_csv("data/Customer Purchasing Behaviors.csv")

df.drop("user_id", axis=1, inplace=True)

# Encode region
le = LabelEncoder()
df["region"] = le.fit_transform(df["region"])

# Save encoder (IMPORTANT for app)
joblib.dump(le, "model/region_encoder.pkl")

# Features & target
X = df.drop("purchase_frequency", axis=1)
y = df["purchase_frequency"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "model/scaler.pkl")

# 🔥 Better model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluation
preds = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, preds))
print("R2 Score:", r2_score(y_test, preds))

# Save model
joblib.dump(model, "model/model.pkl")

print("✅ Model trained and saved!")