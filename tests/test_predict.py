# test_predict.py

"""
test_predict.py — Unit Tests for src/predict.py
Run: python tests/test_predict.py
"""
import pandas as pd
from src.predict import (
    load_artifacts,
    preprocess_input,
    predict_churn,
    get_shap_factors,
    get_risk_level,
    run_prediction,
)

# High-risk customer used across multiple tests
SAMPLE_CUSTOMER = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "No",
    "Dependents": "No", "tenure": 3, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0, "TotalCharges": 255.0
}

preprocessor, model, shap_explainer, metadata = load_artifacts()
feature_names = metadata.get('feature_names', [])
threshold = metadata.get('best_threshold', 0.5)


def test_load_artifacts():
    assert preprocessor is not None, "preprocessor.pkl failed to load"
    assert model is not None, "churn_model.pkl failed to load"
    assert shap_explainer is not None, "shap_explainer.pkl failed to load"
    assert metadata is not None, "metadata failed to load"
    print("✅ test_load_artifacts PASSED")


def test_preprocess_returns_correct_shape():
    processed = preprocess_input(SAMPLE_CUSTOMER, preprocessor)

    if feature_names:
        expected_cols = len(feature_names)
        assert processed.shape == (1, expected_cols), f"Expected shape (1, {expected_cols}) got {processed.shape}"

    print(f"✅ test_preprocess_returns_correct_shape PASSED — shape: {processed.shape}")
    print(processed)
    print(pd.DataFrame(processed))


def test_predict_churn_valid_output():

    processed = preprocess_input(SAMPLE_CUSTOMER, preprocessor)
    prob, risk_level = predict_churn(processed, model, threshold)

    assert 0.0 <= prob <= 1.0, f"Probability out of [0,1] range: {prob}"

    assert risk_level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"], f"Invalid risk level: {risk_level}"

    print(
        f"✅ test_predict_churn_valid_output PASSED — "
        f"P(churn)={prob:.4f}, Risk={risk_level}"
    )


def test_shap_factors_structure():
    processed = preprocess_input(SAMPLE_CUSTOMER, preprocessor)
    factors = get_shap_factors(processed, shap_explainer, feature_names)

    assert len(factors) == 3, f"Expected 3 factors, got {len(factors)}"
    print(factors[0].keys())
    print(f"✅ test_shap_factors_structure PASSED — {len(factors)} factors, all keys present")


def test_get_risk_level_buckets():
    assert get_risk_level(0.85, threshold) == "CRITICAL", "0.85 should be CRITICAL"
    assert get_risk_level(0.60, threshold) == "HIGH",     "0.60 should be HIGH"
    assert get_risk_level(0.45, threshold) == "MEDIUM",   "0.45 should be MEDIUM"
    assert get_risk_level(0.20, threshold) == "LOW",      "0.20 should be LOW"

    print("✅ test_get_risk_level_buckets PASSED — all 4 buckets + edge case verified")


def test_run_prediction_output_schema():
    result = run_prediction(SAMPLE_CUSTOMER)

    required_keys = [
        "churn_probability",
        "risk_level",
        "top_factors",
        "business_impact",
        "recommended_action",
        "explanation",
    ]

    for key in required_keys:
        assert key in result, f"Missing key in output: '{key}'"

    assert isinstance(result["churn_probability"], float)
    assert 0.0 <= result["churn_probability"] <= 1.0

    # top_factors must be a non-empty list
    assert isinstance(result["top_factors"], list) and len(result["top_factors"]) > 0

    print("✅ test_run_prediction_output_schema PASSED — all keys present, types correct")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  UNIT TESTS — src/predict.py")
    print("=" * 60)

    tests = [
        test_load_artifacts,
        test_preprocess_returns_correct_shape,
        test_predict_churn_valid_output,
        test_shap_factors_structure,
        test_get_risk_level_buckets,
        test_run_prediction_output_schema,
    ]

    passed, failed = 0, 0

    for test_fn in tests:
        print(f"\n  Running {test_fn.__name__}...")
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  💥 ERROR: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed  |  {failed} failed")
    print("=" * 60 + "\n")