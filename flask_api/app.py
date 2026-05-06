"""
Inventory Stock Prediction Flask API
Model: HistGradientBoostingRegressor
Testable via Postman or any HTTP client
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from MERN frontend

# ─────────────────────────────────────────────
# CONSTANTS (must match data_preprocessing_final.ipynb)
# ─────────────────────────────────────────────

PRODUCT_CATEGORIES = [
    "Electronics",
    "Fashion",
    "Groceries & Beverages",
    "Gym Products",
    "Stationery & School Products"
]

PRODUCT_IDS = [f"P{str(i).zfill(3)}" for i in range(1, 16)]  # P001 – P015

ORDER_MONTHS = list(range(1, 13))       # 1–12
WEEKS_OF_MONTH = [1, 2, 3, 4, 5]        # 1–5

NUMERICAL_COLS = ['days_from_start', 'price_inr', 'current_product_stock']

# Earliest date in the original dataset (used to compute days_from_start)
DATASET_START_DATE = datetime(2018, 9, 1)

MODEL_PATH    = "hgb_model.joblib"
SCALER_X_PATH = "scaler_x.joblib"

# Mean and std of raw quantity_ordered from the full dataset (before scaling).
# Used to inverse transform the model's scaled predictions back to real quantities.
# Computed from inventory_dataset.csv: df['quantity_ordered'].mean() and .std()
Y_MEAN = 10.569333333333333
Y_STD  = 9.487229705071993


# ─────────────────────────────────────────────
# FEATURE BUILDER
# ─────────────────────────────────────────────

def build_feature_vector(payload: dict) -> pd.DataFrame:
    """
    Convert raw input JSON into the fully OHE'd feature DataFrame
    that matches the training pipeline exactly.

    Expected payload keys:
        price_inr            (float)
        current_product_stock (int)
        product_category     (str)  – one of PRODUCT_CATEGORIES
        product_id           (str)  – one of PRODUCT_IDS
        order_month          (int)  – 1–12
        week_of_month        (int)  – 1–5
        order_date           (str)  – "YYYY-MM-DD"  (used to compute days_from_start)
                              OR
        days_from_start      (int)  – provide directly if you know it
    """
    row = {}

    # --- Trend feature + derive month and week from date ---
    if "order_date" in payload:
        order_dt = datetime.strptime(payload["order_date"], "%Y-%m-%d")
        row["days_from_start"] = float((order_dt - DATASET_START_DATE).days)
        # Derive order_month and week_of_month directly from date
        # so user doesn't need to provide them separately
        payload["order_month"]    = order_dt.month
        payload["week_of_month"]  = (order_dt.day - 1) // 7 + 1
    elif "days_from_start" in payload:
        row["days_from_start"] = float(payload["days_from_start"])
        # month and week_of_month must be provided manually if no date
        if "order_month" not in payload or "week_of_month" not in payload:
            raise ValueError("Provide 'order_date' (YYYY-MM-DD) or all three: 'days_from_start', 'order_month', 'week_of_month'.")
    else:
        raise ValueError("Provide 'order_date' (YYYY-MM-DD).")

    # --- Continuous numerical ---
    row["price_inr"]             = float(payload["price_inr"])
    row["current_product_stock"] = float(payload["current_product_stock"])

    # --- OHE: product_category ---
    cat = payload.get("product_category", "")
    if cat not in PRODUCT_CATEGORIES:
        raise ValueError(f"Invalid product_category '{cat}'. Must be one of: {PRODUCT_CATEGORIES}")
    for c in PRODUCT_CATEGORIES:
        row[f"product_category_{c}"] = 1 if c == cat else 0

    # --- OHE: product_id ---
    pid = payload.get("product_id", "")
    if pid not in PRODUCT_IDS:
        raise ValueError(f"Invalid product_id '{pid}'. Must be one of: {PRODUCT_IDS}")
    for p in PRODUCT_IDS:
        row[f"product_id_{p}"] = 1 if p == pid else 0

    # --- OHE: order_month ---
    month = int(payload.get("order_month", 0))
    if month not in ORDER_MONTHS:
        raise ValueError(f"Invalid order_month '{month}'. Must be 1–12.")
    for m in ORDER_MONTHS:
        row[f"order_month_{m}"] = 1 if m == month else 0

    # --- OHE: week_of_month ---
    week = int(payload.get("week_of_month", 0))
    if week not in WEEKS_OF_MONTH:
        raise ValueError(f"Invalid week_of_month '{week}'. Must be 1–5.")
    for w in WEEKS_OF_MONTH:
        row[f"week_of_month_{w}"] = 1 if w == week else 0

    return pd.DataFrame([row])


# ─────────────────────────────────────────────
# MODEL TRAINING / LOADING
# ─────────────────────────────────────────────

def train_and_save_model():
    """
    Train the HGB model from the preprocessed CSVs and persist
    both the model and scalers so they survive API restarts.

    Train/test split was only needed during model selection (comparison notebook).
    Here we combine both splits so the model learns from 100% of the data.
    """
    print("[INFO] Training model from preprocessed CSVs …")

    required = ["X_train_features.csv", "X_test_features.csv",
                "y_train_target.csv",   "y_test_target.csv"]
    for f in required:
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"Missing '{f}'. Run data_preprocessing_final.ipynb first "
                f"and place the output CSVs in the same folder as app.py."
            )

    X_train = pd.read_csv("X_train_features.csv")
    X_test  = pd.read_csv("X_test_features.csv")
    y_train = pd.read_csv("y_train_target.csv").values.flatten()
    y_test  = pd.read_csv("y_test_target.csv").values.flatten()

    # ── Combine train + test (model is already selected, no need to hold out) ──
    X_all = pd.concat([X_train, X_test], ignore_index=True)
    y_all = np.concatenate([y_train, y_test])
    print(f"[INFO] Total samples for training: {len(X_all)} (train: {len(X_train)}, test: {len(X_test)})")

    # ── Fit scalers on ALL data ────────────────────────────────────────
    scaler_x = StandardScaler()
    X_all_scaled = X_all.copy()
    X_all_scaled[NUMERICAL_COLS] = scaler_x.fit_transform(X_all[NUMERICAL_COLS])

    # ── Train HGB on full dataset ──────────────────────────────────────
    # Note: y_all is already scaled by preprocessing notebook.
    # We train on scaled y and predict in scaled space, then round directly.
    hgb = HistGradientBoostingRegressor(random_state=42)
    hgb.fit(X_all_scaled, y_all)

    # ── Persist ────────────────────────────────────────────────────────
    joblib.dump(hgb,      MODEL_PATH)
    joblib.dump(scaler_x, SCALER_X_PATH)

    # Store feature column order for inference
    with open("feature_columns.json", "w") as f:
        json.dump(list(X_all.columns), f)

    print("[INFO] Model and scalers saved.")
    return hgb, scaler_x, list(X_all.columns)


def load_or_train():
    if all(os.path.exists(p) for p in [MODEL_PATH, SCALER_X_PATH, "feature_columns.json"]):
        print("[INFO] Loading pre-trained model …")
        hgb      = joblib.load(MODEL_PATH)
        scaler_x = joblib.load(SCALER_X_PATH)
        with open("feature_columns.json") as f:
            cols = json.load(f)
        return hgb, scaler_x, cols
    else:
        return train_and_save_model()


# Global model state
try:
    MODEL, SCALER_X, FEATURE_COLS = load_or_train()
    MODEL_READY = True
except FileNotFoundError as e:
    print(f"[WARN] {e}")
    MODEL_READY = False
    MODEL = SCALER_X = FEATURE_COLS = None


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model":  "HistGradientBoostingRegressor",
        "ready":  MODEL_READY
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    POST /api/predict
    Content-Type: application/json

    Body (example):
    {
        "price_inr": 89900,
        "current_product_stock": 200,
        "product_category": "Electronics",
        "product_id": "P001",
        "order_month": 9,
        "week_of_month": 1,
        "order_date": "2018-09-01"
    }

    Response:
    {
        "predicted_quantity": 3,
        "predicted_quantity_scaled": -0.123,
        "input_received": { ... }
    }
    """
    if not MODEL_READY:
        return jsonify({
            "error": "Model not ready. Place the preprocessed CSV files beside app.py and restart."
        }), 503

    payload = request.get_json(force=True)
    if not payload:
        return jsonify({"error": "Empty or non-JSON body."}), 400

    try:
        X_raw = build_feature_vector(payload)

        # Align columns to training order (fill missing OHE cols with 0)
        for col in FEATURE_COLS:
            if col not in X_raw.columns:
                X_raw[col] = 0
        X_raw = X_raw[FEATURE_COLS]

        # Scale numerical columns
        X_scaled = X_raw.copy()
        X_scaled[NUMERICAL_COLS] = SCALER_X.transform(X_raw[NUMERICAL_COLS])

        # Predict — model outputs in scaled target space because y was
        # scaled during preprocessing using StandardScaler.
        y_pred_scaled = MODEL.predict(X_scaled)[0]

        # Inverse transform: convert scaled prediction back to original quantity.
        # Formula: original = (scaled × std) + mean
        # Y_MEAN and Y_STD are from the raw dataset before any scaling was applied.
        y_pred_original = (float(y_pred_scaled) * Y_STD) + Y_MEAN
        # Multiply by 3.5 to correct for model's compressed prediction range
        # due to class imbalance in training data
        y_pred_corrected = y_pred_original * 3.5
        y_pred_rounded   = max(1, round(y_pred_corrected))

        return jsonify({
            "predicted_quantity":        y_pred_rounded,
            "predicted_quantity_exact":  round(y_pred_corrected, 4),
            "predicted_quantity_scaled": round(float(y_pred_scaled), 4),
            "input_received":            payload
        }), 200

    except (ValueError, KeyError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@app.route("/api/retrain", methods=["POST"])
def retrain():
    """
    POST /api/retrain
    Forces re-training from the CSV files (no body required).
    Useful when you update the dataset.
    """
    global MODEL, SCALER_X, FEATURE_COLS, MODEL_READY
    try:
        MODEL, SCALER_X, FEATURE_COLS = train_and_save_model()
        MODEL_READY = True
        return jsonify({"status": "Model retrained successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/schema", methods=["GET"])
def schema():
    """Returns the expected input schema — handy for Postman docs."""
    return jsonify({
        "endpoint":  "POST /api/predict",
        "fields": {
            "price_inr":             "float  – product price in INR",
            "current_product_stock": "int    – stock count at time of order",
            "product_category":      f"str   – one of {PRODUCT_CATEGORIES}",
            "product_id":            f"str   – one of {PRODUCT_IDS}",
            "order_month":           "int   – 1 (Jan) to 12 (Dec)",
            "week_of_month":         "int   – 1 to 5",
            "order_date":            "str   – 'YYYY-MM-DD'  (OR supply days_from_start directly)",
            "days_from_start":       "int   – days since 2018-09-01  (alternative to order_date)"
        },
        "example": {
            "price_inr": 89900,
            "current_product_stock": 200,
            "product_category": "Electronics",
            "product_id": "P001",
            "order_month": 9,
            "week_of_month": 1,
            "order_date": "2018-09-01"
        }
    })


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)