import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template
BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
model = None
scaler = None
feature_columns = None

def load_models():
    global model, scaler, feature_columns
    try:
        model = joblib.load(str(ASSETS_DIR / "food_waste_model.pkl"))
        scaler = joblib.load(str(ASSETS_DIR / "scaler.pkl"))
        feature_columns = joblib.load(str(ASSETS_DIR / "features.pkl"))
        print(f"[CrumbIQ] Models loaded successfully.", flush=True)
        print(f"[CrumbIQ] Feature columns: {list(feature_columns)}", flush=True)
    except Exception as e:
        print(f"[CrumbIQ] ERROR loading models: {e}", flush=True)
        sys.exit(1)
load_models()
SECTION_MAP = {
    "Section A": "A",
    "Section B": "B",
    "Section C": "C",
    "Section D": "D",
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D",
}

CATEGORY_MAP = {
    "Grains":     "Rice",
    "Proteins":   "Meat",
    "Vegetables": "Vegetables",
    "Fruits":     "Vegetables",
    "Dairy":      "Soup",
    "Desserts":   "Meat",
    "Rice":       "Rice",
    "Meat":       "Meat",
    "Soup":       "Soup",
}

UNIT_PRICE_MAP = {
    "Rice":       2.0,
    "Vegetables": 3.0,
    "Soup":       1.5,
    "Meat":       8.0,
}

MEAL_MAP = {
    "Breakfast": "Breakfast",
    "Lunch":     "Lunch",
    "Dinner":    "Dinner",
    "Snacks":    "Breakfast",
}

RICHER_RECOMMENDATIONS = {
    "Grains": {
        "low":    "Grain and rice volumes are optimal. Continue current preparation ratios.",
        "medium": "Consider reducing rice cooking sheets by 12–15%. Prepare grains in batches to avoid premature cooling waste.",
        "high":   "Significantly over-preparing grains. Reduce rice prep by 25–30% and hold reserves as dry stock. Alert kitchen supervisor.",
    },
    "Proteins": {
        "low":    "Protein preparation is well-calibrated. No adjustment required.",
        "medium": "Scale down meat and lentil portions by 10–15%. Introduce smaller batch cooking cycles every 45 minutes.",
        "high":   "High protein waste detected. Reduce main entrée cooking volume by 30%. Switch to staggered cooking to prevent surplus.",
    },
    "Vegetables": {
        "low":    "Vegetable sides are efficiently portioned. Maintain current prep schedule.",
        "medium": "Trim vegetable side dish quantities by 15%. Pre-prep only first-batch volumes and hold remainder as raw inventory.",
        "high":   "Severe vegetable overproduction. Cut side dish prep by 30%. Reassign excess pre-cut vegetables to breakfast service.",
    },
    "Fruits": {
        "low":    "Fruit portions are within acceptable bounds. Keep pre-sliced volumes steady.",
        "medium": "Cap fruit platters at 60% of current portions. Serve whole fruits on demand to reduce pre-cut spoilage.",
        "high":   "Critical fruit surplus risk. Limit pre-cut fruit to demand-only service. Store remaining inventory in cold storage.",
    },
    "Dairy": {
        "low":    "Dairy distribution is optimal. Continue standard portioning.",
        "medium": "Reduce dairy pre-distribution by 15%. Hold milk and curd reserves in cold storage until needed.",
        "high":   "High dairy waste predicted. Limit counter deployment to 50% of normal. Log remaining stock for next service reuse.",
    },
    "Desserts": {
        "low":    "Dessert quantities are well-matched to expected demand.",
        "medium": "Scale down dessert display by 20%. Serve high-spoilage items first to minimize end-of-service discard.",
        "high":   "Major dessert surplus expected. Reduce preparation by 35%. Consider switching to longer shelf-life items for this service.",
    },
}

CAFETERIA_CAPACITY = 275.0

def get_risk_level(waste_kg: float) -> str:
    if waste_kg < 2.0:
        return "low"
    elif waste_kg < 3.5:
        return "medium"
    else:
        return "high"

def get_risk_label(waste_kg: float) -> str:
    level = get_risk_level(waste_kg)
    return {
        "low":    "Low Waste Risk",
        "medium": "Moderate Waste Risk",
        "high":   "High Waste Risk",
    }[level]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if data is None:
            return jsonify({"error": "No JSON body received"}), 400

        raw_meal      = data.get("Meal", "Lunch")
        raw_section   = data.get("Canteen_Section", "A")
        raw_category  = data.get("Food_Category", "Vegetables")
        attendance    = int(data.get("attendance", CAFETERIA_CAPACITY))
        date_str      = data.get("Date", "")

        meal     = MEAL_MAP.get(raw_meal, raw_meal)
        section  = SECTION_MAP.get(raw_section, "A")
        category = CATEGORY_MAP.get(raw_category, "Vegetables")

        unit_price = UNIT_PRICE_MAP.get(category, 3.0)

        input_dict = {
            "Meal":                meal,
            "Canteen_Section":     section,
            "Food_Category":       category,
            "Unit_Price_per_kg":   unit_price,
        }

        if date_str:
            try:
                dt = pd.to_datetime(date_str)
                input_dict["DayOfWeek"]  = dt.dayofweek
                input_dict["Month"]      = dt.month
                input_dict["Day"]        = dt.day
            except Exception:
                pass

        input_df = pd.DataFrame([input_dict])
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        input_scaled   = scaler.transform(input_df)
        base_prediction = model.predict(input_scaled)[0]

        attendance_factor = attendance / CAFETERIA_CAPACITY
        prediction = float(base_prediction) * attendance_factor
        prediction = max(prediction, 0.1)

        RUPEES_PER_KG = 45.0
        estimated_loss = round(prediction * RUPEES_PER_KG * unit_price, 2)

        risk_level = get_risk_level(prediction)
        risk_label = get_risk_label(prediction)

        recommendation = RICHER_RECOMMENDATIONS.get(raw_category, {}).get(
            risk_level,
            "Monitor portion sizes and adjust preparation volumes as needed."
        )

        return jsonify({
            "predicted_waste":  round(prediction, 2),
            "estimated_loss":   round(estimated_loss, 2),
            "risk":             risk_label,
            "risk_level":       risk_level,
            "recommendation":   recommendation,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
