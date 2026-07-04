# api/main.py

import os
import sys
import logging
from collections import Counter
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from dotenv import load_dotenv

# Make src/ importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GROQ_MODEL = "llama-3.1-8b-instant"  # Groq cloud model
from src.predict import run_prediction

from api.schemas import (
    CustomerInput, PredictionResponse,
    WhatIfInput, WhatIfResponse, BatchInput, BatchSummary, BatchPredictionResponse
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s:  %(message)s")
logger = logging.getLogger(__name__)

# Global Grok client — initialized once at startup, reused on every request
groq_client = None


# API lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
        Code before `yield` runs at startup.
        Code after `yield` runs at shutdown.
    """
    global groq_client

    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        groq_client = Groq(api_key=api_key)
        logger.info("   ✅ Groq client initialized")
    else:
        logger.warning("⚠️ API_KEY not set — AI explanations will use fallback")

    yield  # API is live and serving requests here

    logger.info("   API shutting down")


app = FastAPI(
    title="Customer Churn Predictor API",
    description="""
    Predict customer churn with XGBoost + SHAP explainability + Groq NL explanations
    
    Endpoints
    - POST /predict** — Single customer risk score with explanation
    - POST /predict/whatif** — Simulate retention interventions
    - POST /predict/batch** — Bulk score up to 100 customers
    - GET /health** — Service liveness check
    
    Differentiators
    - SHAP-based explainability (not a black box)
    - Business-framed output (revenue at risk, not just accuracy)
    - What-If simulator for retention strategy testing
    - Batch endpoint for CRM integration
    """,
    version="1.0.0",
    lifespan=lifespan,
)

"""
CORS (Cross-Origin Resource Sharing) lets a React frontend or any web app on a different domain call your API. 
Without it, browser-based clients get blocked."""
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


def get_grok_explanation(prediction: dict, customer: dict) -> str:
    """
    Sends the ML prediction + customer profile to Grok.
    Returns a 2-3 sentence plain-English explanation.
    Falls back to a template if Grok is unavailable.
    :param prediction: dict
    :param customer: dict
    :return: plain english prediction
    """
    if groq_client is None:
        return _fallback_explanation(prediction)

    try:
        factors_str = "\n".join([
            f"{f['feature']}: {f['direction']} (SHAP: {f['impact']})"
            for f in prediction['top_factors']
        ])

        prompt = f"""You are a customer retention analyst at a telecom company.
        The ML model has scored this customer:
        
        Churn Probability: {prediction['churn_probability']:.1%}
        Risk Level: {prediction['risk_level']}
        
        Top 3 SHAP drivers:
        {factors_str}
        
        Customer snapshot:
        - Tenure: {customer.get('tenure')} months with the service
        - Contract type: {customer.get('Contract')}
        - Monthly charges: ₹{customer.get('MonthlyCharges')}
        - Internet service: {customer.get('InternetService')}
        - Online Security add-on: {customer.get('OnlineSecurity')}
        
        Write 2-3 sentences explaining WHY this customer is at {prediction['risk_level']} risk of churning.
        Rules:
        - Plain business language only, zero ML jargon
        - Be specific about the actual drivers above
        - Write for a retention agent who will use this to make a call
        - Under 70 words"""
        # groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise customer retention analyst. "
                               "Give specific, actionable insights in plain English."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=120,
            temperature=0.3,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Grok API call failed: {e}")
        return _fallback_explanation(prediction)


def _fallback_explanation(prediction: dict) -> str:
    """Template used when Grok is unavailable — API still works."""
    prob = prediction["churn_probability"]
    risk = prediction["risk_level"]
    factors = prediction.get("top_factors", [])
    driver = factors[0]["feature"] if factors else "multiple factors"
    return (
        f"Model predicts {prob:.1%} churn probability ({risk} risk). "
        f"Primary driver: '{driver}'. "
        f"See top_factors for the complete breakdown."
    )


# health endpoint
@app.get("/health", tags=["Status"])
async def health():
    """
    Standard health check. Every production API has this.
    Render, Kubernetes, and load balancers ping this to check if the app is alive.
    """
    return {
        "status": "healthy",
        "grok_enabled": groq_client is not None,
        "version": "1.0.0",
    }


# predict endpoint
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(customer: CustomerInput):
    """
    predict churn probability for one customer.
    :param customer: customer dict
    :return: probability, risk level, top 3 SHAP factors,
    business impact framing, recommended action, and
    a Grok-generated plain-English explanation.
    """
    try:
        customer_dict = customer.model_dump()  # Pydantic → plain Python dict

        result = run_prediction(customer_dict)
        # result['churn_probability'] = "{.1%}".format(result['churn_probability'])
        result['explanation'] = get_grok_explanation(result, customer_dict)

        return result

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# what-if endpoint
@app.post("/predict/whatif", response_model=WhatIfResponse, tags=["Prediction"])
async def predict_whatif(request: WhatIfInput):
    """
    What-If Simulator: quantify the ROI(return of investment) of a retention intervention.

    Scenario example:
    'If I upgrade this customer from Month-to-month to a Two-year contract
     AND add Online Security, how much does churn risk drop?
     What is the annual revenue I'm protecting?'
    A CRM can call this before every retention agent call.
    :param request:og customer dict + proposed changes
    :return:post changes probability
    """
    try:
        # current state
        current_dict = request.customer.model_dump()
        current_result = run_prediction(current_dict)
        current_result['explanation'] = get_grok_explanation(current_result, current_dict)

        # proposed state
        proposed_dict = current_dict.copy()
        proposed_dict.update(request.proposed_change)
        proposed_result = run_prediction(proposed_dict)
        proposed_result['explanation'] = get_grok_explanation(proposed_result, proposed_dict)

        # Business value calculation
        prob_before = current_result['churn_probability']
        prob_after = proposed_result['churn_probability']
        risk_reduction_pct = (prob_before - prob_after) * 100

        monthly_charges = current_dict.get('MonthlyCharges', 0)
        retained_prob = max(prob_before - prob_after, 0)
        annual_value = retained_prob * monthly_charges * 12

        if risk_reduction_pct > 0:
            intervention_value = (
                f"Churn risk drops by {risk_reduction_pct:.1f} percentage points. "
                f"Expected annual revenue protected: ₹{annual_value:,.0f} "
                f"(Δ{retained_prob:.1%} × ₹{monthly_charges}/month × 12 months)."
            )
            recommendation = (
                f"✅ Recommend this intervention. "
                f"At scale across 100 similar customers: "
                f"₹{annual_value * 100:,.0f}/year in revenue protected."
            )
        else:
            intervention_value = (
                f"This change increases churn risk by {abs(risk_reduction_pct):.1f}pp. "
                "Not recommended."
            )
            recommendation = (
                "❌ This intervention worsens churn risk. "
                "Try a different change."
            )
        return WhatIfResponse(
            current=PredictionResponse(**current_result),
            proposed=PredictionResponse(**proposed_result),
            risk_reduction_pct=round(risk_reduction_pct, 2),
            intervention_value=intervention_value,
            recommendation=recommendation,
        )

    except Exception as e:
        logger.error(f"What-If error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# predict/batch endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(batch: BatchInput):
    """
    Score up to 100 customers in a single request
    """
    predictions = []
    errors = []

    for i, customer in enumerate(batch.customers):
        try:
            customer_dict = customer.model_dump()
            result = run_prediction(customer_dict)

            # Only call Groq if client explicitly requested explanations
            # Default: skip Groq → 100x faster batch processing
            if batch.include_explanations:
                explanation = get_grok_explanation(result, customer_dict)
            else:
                explanation = _fallback_explanation(result)

            result['explanation'] = explanation
            predictions.append(result)

        except Exception as e:
            # execution must be continued even thought one bas customer encountered
            logger.error(f"Batch error on customer index {i} (id = {customer.customer_id}: {e}")
            errors.append({"index": i, "customer_id": customer.customer_id, "error": e})

    if not predictions:
        raise HTTPException(status_code=500, detail="All customers in batch failed processing")

    # executive summary

    critical_risk = [
        (p, c) for p, c in zip(predictions, batch.customers) if p['risk_level'] == 'CRITICAL'
    ]
    high_risk = [
        (p, c) for p, c in zip(predictions, batch.customers) if p['risk_level'] == 'HIGH'
    ]
    medium_risk = [p for p in predictions if p['risk_level'] == "MEDIUM"]
    low_risk = [p for p in predictions if p['risk_level'] == "LOW"]

    # Average churn probability across all customers
    avg_prob = sum(p['churn_probability'] for p in predictions) / len(predictions)

    # Revenue at risk = sum(MonthlyCharges × 12) for HIGH risk customers only
    # Logic: if they churn, we lose their annual spend
    # Using input charges (not from prediction result — more reliable)
    critical_risk_charges = [c.MonthlyCharges for _, c in critical_risk]
    high_risk_charges = [c.MonthlyCharges for _, c in high_risk]
    total_revenue_at_risk = (sum(charge * 12 for charge in critical_risk_charges)
                             + sum(charge * 12 for charge in high_risk_charges))

    # Top churn drivers: collect top-2 SHAP features per customer, find most common
    all_drivers = []
    for p in predictions:
        all_drivers.extend([f['feature'] for f in p['top_factors'][:2]])
    top_drivers = [driver for driver, _ in Counter(all_drivers).most_common(5)]

    summary = BatchSummary(
        total_customers=len(predictions),
        critical_risk_count=len(critical_risk),
        high_risk_count=len(high_risk),
        medium_risk_count=len(medium_risk),
        low_risk_count=len(low_risk),
        average_churn_probability=round(avg_prob, 4),
        total_annual_revenue_at_risk=round(total_revenue_at_risk, 2),
        top_churn_drivers=top_drivers
    )

    logger.info(
            f"Batch complete: {len(predictions)} | "
            f"High risk: {len(high_risk) + len(critical_risk)} | Revenue at risk: ${total_revenue_at_risk:,.2f}"
        )

    return BatchPredictionResponse(
        predictions=predictions,
        summary=summary
    )


if __name__ == "__main__":
    customer_dict = {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 3, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.5, "TotalCharges": 286.5
    }
    print(get_grok_explanation(run_prediction(customer_dict), customer_dict))
