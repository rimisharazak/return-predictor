import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "artifacts", "model_data.joblib")

app = Flask(__name__)

_model_data = None


def load_model():
    """Loads artifacts/model_data.joblib (the dict you built with joblib.dump
    at the end of your notebook: model, scaler, model_columns, numeric_features,
    binary_flags, categorical_features, train_medians, metrics, feature_importance,
    threshold)."""
    global _model_data
    if _model_data is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Model file not found at artifacts/model_data.joblib. "
                "Copy the model_data.joblib file produced by your notebook "
                "(joblib.dump(model_data, 'artifacts/model_data.joblib')) into "
                "this project's artifacts/ folder, then restart the server."
            )
        _model_data = joblib.load(MODEL_PATH)
    return _model_data


@app.route("/")
def index():
    model_data = None
    error = None
    try:
        model_data = load_model()
    except FileNotFoundError as e:
        error = str(e)

    context = {"error": error, "model_name": "Logistic Regression", "metrics": {}, "feature_importance": []}
    if model_data:
        context["model_name"] = model_data.get("model_name", "Model")
        context["metrics"] = model_data.get("metrics", {})
        context["feature_importance"] = model_data.get("feature_importance", [])[:10]
    return render_template("index.html", **context)


def build_feature_row(payload, model_data):
    numeric_features = model_data["numeric_features"]
    binary_flags = model_data["binary_flags"]
    categorical_features = model_data["categorical_features"]
    train_medians = model_data["train_medians"]
    model_columns = model_data["model_columns"]
    scaler = model_data["scaler"]

    def as_float(key, default=0.0):
        val = payload.get(key, default)
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    price = as_float("price", 0.0)
    support_calls = as_float("support_calls", 0.0)
    chat_messages = as_float("chat_messages", 0.0)

    row = {
        "product_category": payload.get("product_category"),
        "sub_category": payload.get("sub_category"),
        "brand": payload.get("brand"),
        "discount_percent": as_float("discount_percent"),
        "product_rating": as_float("product_rating"),
        "review_count": as_float("review_count"),
        "fragile_item": int(bool(payload.get("fragile_item"))),
        "warranty_available": int(bool(payload.get("warranty_available"))),
        "product_return_rate": as_float("product_return_rate"),
        "category_return_rate": as_float("category_return_rate"),
        "brand_return_rate": as_float("brand_return_rate"),
        "defect_rate": as_float("defect_rate"),
        "seller_rating": as_float("seller_rating"),
        "seller_return_rate": as_float("seller_return_rate"),
        "fulfillment_type": payload.get("fulfillment_type"),
        "payment_method": payload.get("payment_method"),
        "quantity": as_float("quantity", 1.0),
        "shipping_distance_km": as_float("shipping_distance_km"),
        "delayed_delivery": int(bool(payload.get("delayed_delivery"))),
        "wishlist_before_purchase": int(bool(payload.get("wishlist_before_purchase"))),
        "product_page_views": as_float("product_page_views"),
        "total_support_contacts": support_calls + chat_messages,
        "log_price": float(np.log1p(price)),
    }

    df = pd.DataFrame([row])

    # 1. Impute any missing values with the training medians (mirrors notebook step)
    all_num = numeric_features + binary_flags
    for col in all_num:
        if col in train_medians and (pd.isna(df.at[0, col])):
            df.at[0, col] = train_medians[col]

    # 2. Scale numeric features with the fitted StandardScaler
    df[numeric_features] = scaler.transform(df[numeric_features])

    # 3. One-hot encode categoricals, then align to the training columns
    df = pd.get_dummies(df, columns=categorical_features, dtype=int)
    df = df.reindex(columns=model_columns, fill_value=0)

    return df


@app.route("/predict", methods=["POST"])
def predict():
    try:
        model_data = load_model()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400

    payload = request.get_json(force=True) or {}
    try:
        X = build_feature_row(payload, model_data)
        model = model_data["model"]
        proba = float(model.predict_proba(X)[0][1])
        threshold = float(model_data.get("threshold", 0.5))
        will_return = proba >= threshold

        if proba >= 0.66:
            risk_level = "High Risk"
        elif proba >= 0.33:
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"

        return jsonify({
            "probability": round(proba * 100, 1),
            "will_return": bool(will_return),
            "risk_level": risk_level,
        })
    except KeyError as e:
        return jsonify({"error": f"Missing field or model artifact key: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0', port=3000)
