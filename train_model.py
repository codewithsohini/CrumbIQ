"""
CrumbIQ — Retrain RandomForest model on the cafeteria waste dataset.
Outputs: food_waste_model.pkl, scaler.pkl, features.pkl
placed in attached_assets/ to replace the survey-trained versions.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

BASE_DIR = Path(__file__).parent
ASSETS   = BASE_DIR.parent / "attached_assets"
DATASET  = ASSETS / "Dataset_Propely_1780062567156.csv"

print("[TRAIN] Loading dataset …")
df = pd.read_csv(DATASET)
print(f"[TRAIN] Shape: {df.shape}")
print(f"[TRAIN] Columns: {list(df.columns)}")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date", "Waste_Weight_kg"])

df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Month"]     = df["Date"].dt.month
df["Day"]       = df["Date"].dt.day

TARGET   = "Waste_Weight_kg"
DROP_COLS = ["Date", TARGET, "Cost_Loss"]
feature_df = df.drop(columns=DROP_COLS, errors="ignore")

feature_df = pd.get_dummies(feature_df)
y = df[TARGET].values

print(f"[TRAIN] Features after encoding: {list(feature_df.columns)}")

X_train, X_test, y_train, y_test = train_test_split(
    feature_df, y, test_size=0.2, random_state=42
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae   = mean_absolute_error(y_test, preds)
r2    = r2_score(y_test, preds)
print(f"[TRAIN] MAE={mae:.4f}  R²={r2:.4f}")

joblib.dump(model,                  ASSETS / "food_waste_model_1780062567161.pkl")
joblib.dump(scaler,                 ASSETS / "scaler_1780062567164.pkl")
joblib.dump(list(feature_df.columns), ASSETS / "features_1780062567159.pkl")

print("[TRAIN] Saved: food_waste_model.pkl  scaler.pkl  features.pkl")
print("[TRAIN] Feature columns:", list(feature_df.columns))
