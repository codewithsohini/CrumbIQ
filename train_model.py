"""
CrumbIQ — Retrain RandomForest model on the cafeteria waste dataset.
Outputs:
    food_waste_model.pkl
    scaler.pkl
    features.pkl
"""

from pathlib import Path
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).parent

DATASET = BASE_DIR / "Dataset_Propely.csv"
# =========================
# LOAD DATA
# =========================

print("[TRAIN] Loading dataset...")

df = pd.read_csv(DATASET)

print(f"[TRAIN] Shape: {df.shape}")

# =========================
# DATE FEATURES
# =========================

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df = df.dropna(subset=["Date", "Waste_Weight_kg"])

df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

# =========================
# TARGET
# =========================

TARGET = "Waste_Weight_kg"

DROP_COLS = [
    "Date",
    "Waste_Weight_kg",
    "Cost_Loss"
]

X = df.drop(columns=DROP_COLS, errors="ignore")
y = df[TARGET]

# =========================
# ENCODING
# =========================

X = pd.get_dummies(X)

feature_columns = list(X.columns)

print(f"[TRAIN] Total Features: {len(feature_columns)}")

# =========================
# SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# SCALING
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# MODEL
# =========================

model = RandomForestRegressor(
    n_estimators=100,   # reduced size
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# =========================
# EVALUATION
# =========================

preds = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"[TRAIN] MAE: {mae:.4f}")
print(f"[TRAIN] R² : {r2:.4f}")

# =========================
# FEATURE IMPORTANCE
# =========================

feature_importance = sorted(
    zip(feature_columns, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True
)

print("\nTop 10 Features:")

for feature, importance in feature_importance[:10]:
    print(f"{feature}: {importance:.4f}")

# =========================
# SAVE FILES
# =========================

joblib.dump(
    model,
    BASE_DIR / "food_waste_model.pkl",
    compress=5
)

joblib.dump(
    scaler,
    BASE_DIR / "scaler.pkl",
    compress=5
)

joblib.dump(
    feature_columns,
    BASE_DIR / "features.pkl",
    compress=5
)

print("\n[TRAIN] Files saved successfully:")
print(" - food_waste_model.pkl")
print(" - scaler.pkl")
print(" - features.pkl")
