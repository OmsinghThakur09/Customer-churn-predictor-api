# api/main.py

import os
import sys
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

# Make src/ importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GROK_MODEL = "llama-3.1-8b-instant"  # Groq cloud model
from src.predict import run_prediction

from api.schemas import (
    CustomerInput, PredictionResponse,
    WhatIfInput, WhatIfResponse, ShapFactor
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s:  %(message)s")
logger = logging.getLogger(__name__)

# Global Grok client — initialized once at startup, reused on every request
grok_client: OpenAI | None = None


# API lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
        Code before `yield` runs at startup.
        Code after `yield` runs at shutdown.
    """
    global grok_client

    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        grok_client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
        logger.info("✅ Groq client initialized")
    else:
        logger.warning("⚠️ API_KEY not set — AI explanations will use fallback")

    yield  # API is live and serving requests here

    logger.info("API shutting down")


app = FastAPI(
    title="Customer Churn Predictor API",
    description=(
        "Predicts telecom customer churn SHAP factor explanations "
        "and Grok-powered plain-English insights. Built by Om Sanjaysingh Bais."
    ),
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
    if grok_client is None:
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

        response = grok_client.chat.completions.create(
            model=GROK_MODEL,
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
    :return: health stat
    """
    return {
        "status": "healthy",
        "grok_enabled": grok_client is not None,
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

        result['explanation'] = get_grok_explanation(result, customer_dict)

        return result

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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
