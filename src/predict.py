# src/predict.py

import json
import numpy as np
import pandas as pd
import joblib
import shap
from pathlib import Path

# paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"


# print(MODELS_DIR)


# loading models and metadata
def load_artifacts() -> tuple:
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    model = joblib.load(MODELS_DIR / "churn_model.pkl")
    shap_explainer = joblib.load(MODELS_DIR / "shap_explainer.pkl")

    with open(MODELS_DIR / "model_metadata.json", "r") as f:
        metadata = json.load(f)

    return preprocessor, model, shap_explainer, metadata


# preprocessor, _, _, metadata = load_artifacts()
# print(metadata.keys())
# print(metadata)          


# get risk according to probability
def get_risk_level(probability: float, threshold: float) -> str:
    """
    calculating boundaries for risk level decision based on best threshold we are getting from
    ROC curve.
    :param probability:
    :param threshold:
    :return:
    """
    if probability >= 0.85:
        return "CRITICAL"
    elif 0.84 >= probability >= 0.70:
        return "HIGH"
    elif 0.69 >= probability >= 0.49:
        return "MEDIUM"
    else:
        return "LOW"


# preprocess the input customer dict
def preprocess_input(customer_dict: dict, preprocessor) -> np.ndarray:
    df = pd.DataFrame([customer_dict])
    processed = preprocessor.transform(df)
    return processed


# predict probability of churn for input customer
def predict_churn(processed_input: np.ndarray,
                  model,
                  threshold: float) -> tuple:
    proba_array = model.predict_proba(processed_input)

    probability = float(proba_array[0][1])
    risk_level = get_risk_level(probability, threshold)

    return probability, risk_level


# get shap factors for the input customer
def get_shap_factors(
        processed_input: np.ndarray,
        shap_explainer,
        feature_names: list,
        top_n: int = 3
) -> list:
    shap_values = shap_explainer.shap_values(processed_input)

    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    top_indices = np.argsort(np.abs(sv))[::-1][:top_n]

    factors = []
    for idx in top_indices:
        impact = float(sv[idx])
        factors.append({
            "feature": feature_names[idx],
            "impact": f"{impact:+.4f}",
            "direction": "increases churn risk" if impact > 0 else "decreases churn risk"
        })
    return factors


# Business Impact
def calculate_business_impact(probability: float, monthly_charges: float) -> str:
    annual_value = monthly_charges * 12
    expected_loss = probability * annual_value
    return (
        f"Expected revenue at risk: ₹{expected_loss:,.0f}/year "
        f"(₹{monthly_charges:.0f}/month × 12 × {probability:.0%} churn probability)"
    )


def get_recommended_action(risk_level: str) -> str:
    actions = {
        "CRITICAL": "Immediate retention call — offer contract upgrade + 20% discount",
        "HIGH": "Flag for retention team — personalised offer within 48 hours",
        "MEDIUM": "Add to watch list — trigger check-in email campaign",
        "LOW": "No action needed — customer relationship is healthy"
    }
    return actions[risk_level]


def run_prediction(customer_dict: dict) -> dict:
    # step 1
    preprocessor, model, shap_explainer, metadata, = load_artifacts()

    threshold = metadata.get("optimal_threshold", metadata.get("threshold", 0.5))
    feature_names = metadata.get("feature_names", [])
    # backup
    if not feature_names:
        feature_names = pd.read_csv(DATA_DIR / "feature_names.csv").iloc[:, 0].tolist()

    # step 2
    processed = preprocess_input(customer_dict, preprocessor)

    # step 3
    probability, risk_level = predict_churn(processed, model, threshold)

    # step 4
    top_factors = get_shap_factors(processed, shap_explainer, feature_names)

    # step 5
    monthly_charge = customer_dict.get("MonthlyCharges", 0.0)

    return {
        "churn_probability": round(probability, 4),
        "risk_level": risk_level,
        "top_factors": top_factors,
        "business_impact": calculate_business_impact(probability, monthly_charge),
        "recommended_action": get_recommended_action(risk_level),
        "explanation": "← LLM API fills this"
    }


if __name__ == "__main__":

    TEST_CUSTOMERS = [
        {
            "label": "Customer A — Expected: CRITICAL",
            "data": {
                "gender": "Male", "SeniorCitizen": 0, "Partner": "No",
                "Dependents": "No", "tenure": 1, "PhoneService": "Yes",
                "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "No", "StreamingMovies": "No",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 95.0, "TotalCharges": 95.0
            }
        },
        {
            "label": "Customer B — Expected: HIGH",
            "data": {
                "gender": "Female", "SeniorCitizen": 1, "Partner": "No",
                "Dependents": "No", "tenure": 5, "PhoneService": "Yes",
                "MultipleLines": "Yes", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "Yes", "StreamingMovies": "Yes",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.0, "TotalCharges": 425.0
            }
        },
        {
            "label": "Customer C — Expected: MEDIUM",
            "data": {
                "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
                "Dependents": "No", "tenure": 18, "PhoneService": "Yes",
                "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "Yes", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "Yes", "StreamingMovies": "No",
                "Contract": "One year", "PaperlessBilling": "Yes",
                "PaymentMethod": "Credit card (automatic)",
                "MonthlyCharges": 65.0, "TotalCharges": 1170.0
            }
        },
        {
            "label": "Customer D — Expected: LOW",
            "data": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
                "Dependents": "Yes", "tenure": 60, "PhoneService": "Yes",
                "MultipleLines": "Yes", "InternetService": "DSL",
                "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
                "DeviceProtection": "Yes", "TechSupport": "Yes",
                "StreamingTV": "No", "StreamingMovies": "No",
                "Contract": "Two year", "PaperlessBilling": "No",
                "PaymentMethod": "Bank transfer (automatic)",
                "MonthlyCharges": 45.0, "TotalCharges": 2700.0
            }
        },
        {
            "label": "Customer E — Expected: LOW (loyal long-term)",
            "data": {
                "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
                "Dependents": "Yes", "tenure": 72, "PhoneService": "Yes",
                "MultipleLines": "No", "InternetService": "No",
                "OnlineSecurity": "No internet service",
                "OnlineBackup": "No internet service",
                "DeviceProtection": "No internet service",
                "TechSupport": "No internet service",
                "StreamingTV": "No internet service",
                "StreamingMovies": "No internet service",
                "Contract": "Two year", "PaperlessBilling": "No",
                "PaymentMethod": "Bank transfer (automatic)",
                "MonthlyCharges": 22.0, "TotalCharges": 1584.0
            }
        }
    ]

    print("\n" + "=" * 65)
    print("  CUSTOMER CHURN PREDICTION Test Run")
    print("=" * 65)

    for customer in TEST_CUSTOMERS:
        print(f"\n{'─' * 65}")
        print(f"  {customer['label']}")
        print(f"{'─' * 65}")

        result = run_prediction(customer["data"])

        risk_icons = {
            "CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"
        }
        icon = risk_icons.get(result["risk_level"], "⚪")

        print(f"  {icon} Risk Level        : {result['risk_level']}")
        print(f"  📊 Churn Probability : {result['churn_probability']:.1%}")
        print(f"  💰 Business Impact   : {result['business_impact']}")
        print(f"  🎯 Action            : {result['recommended_action']}")
        print(f"\n  Top Factors Driving This Prediction:")
        for i, f in enumerate(result["top_factors"], 1):
            print(f"    {i}. {f['feature']}")
            print(f"       Impact: {f['impact']}  ({f['direction']})")

    print("\n" + "=" * 65)
