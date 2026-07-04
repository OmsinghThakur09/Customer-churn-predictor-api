# Customer Churn Predictor API

**Not a churn-probability toy — a decision-support system for retention teams.**

Predicts which customers are about to cancel, explains *why* in plain English, quantifies the ROI of specific retention offers before anyone picks up the phone, and scores an entire customer base in one API call.

Built by **Om Sanjaysingh Bais**

---

## The Business Problem

Subscription businesses lose recurring revenue quietly. A customer doesn't announce they're about to leave — they just stop renewing. By the time churn shows up in a monthly report, the revenue is already gone.

Most "churn prediction" projects stop at a probability score. That's not useful to a retention team on its own — a retention agent doesn't need "82% churn probability," they need to know **who to call, why they're at risk, and what offer will actually change their mind, before they spend the discount budget finding out.**

This API answers all three.

---

## Why This Isn't "Just Another Churn Predictor"

The Telco Churn dataset is one of the most-repeated projects in ML portfolios — a scan of public repositories turns up hundreds of near-identical entries: EDA notebook, one-hot encoding, a Logistic Regression / XGBoost / LightGBM comparison, a confusion matrix, sometimes a Flask or Streamlit demo. A handful go further with Airflow/Kafka/Spark pipelines aimed at engineering scale, not business usability. Almost none pair the prediction with an LLM-generated explanation, a "what happens if I make this offer" ROI simulator, or a deployed multi-endpoint API a CRM could actually call. That gap is where this project sits deliberately:

| Typical churn project         | This project                                                                                                        |
|-------------------------------|---------------------------------------------------------------------------------------------------------------------|
| Notebook only                 | Deployed REST API                                                                                                   |
| "82% probability"             | "82% probability, driven by month-to-month contract + fiber optic service, ₹930/year at risk, call within 48 hours" |
| Black-box prediction          | SHAP-attributed, per-customer explanation                                                                           |
| No plain-English layer        | Groq LLM (`llama3-8b-8192`) translates SHAP output into a sentence a non-technical account manager can read         |
| No "what to do about it"      | **What-If ROI Simulator** — tests a specific retention offer *before* it's offered                                  |
| Single-customer only          | Batch endpoint scores up to 100 customers in one call, ranked, with an executive summary                            |
| Accuracy-framed               | Revenue-framed — every prediction is tied to ₹ at risk, not just a probability                                      |
| Silent failure if LLM is down | Graceful fallback — the core prediction still returns even if Groq is unreachable                                   |

The differentiator isn't the model — XGBoost on the Telco dataset is a solved problem. The differentiator is that this ships as something a retention team could actually use on Monday morning.

---

## Dataset

**Kaggle Telco Customer Churn** — 7,043 customers, 26.5% churn rate, 19 predictor features (excluding customer ID and the Churn target) covering demographics, account details (tenure, contract, billing), and subscribed services (internet, streaming, security add-ons).

---

## Architecture

```
Raw CSV (Kaggle)
      │
      ▼
preprocess.py ──► ColumnTransformer (OneHotEncoder + StandardScaler + SimpleImputer)
      │            + SMOTE (train set only, applied AFTER split to prevent leakage)
      ▼
      26 engineered features ──► preprocessor.pkl
      │
      ▼
train.py ──► XGBoost + RandomizedSearchCV (5-fold StratifiedKFold, 20 iterations)
      │       Threshold tuned on Precision-Recall curve (not default 0.5)
      │       SHAP TreeExplainer fit on final model
      ▼
      churn_model.pkl + shap_explainer.pkl + model_metadata.json
      │
      ▼
predict.py ──► loads artifacts, runs prediction, extracts SHAP factors,
      │         converts to business-framed output (₹ at risk, risk tier)
      ▼
main.py (FastAPI) ──► 4 endpoints, Groq LLM layer for plain-English explanations
      │
      ▼
Render (deployed, public URL)
```

---

## Model Selection & Performance

Three models were evaluated: **Logistic Regression** (baseline, for interpretable coefficients), **XGBoost**, and **LightGBM**. XGBoost was selected as the final production model based on cross-validated performance on this dataset size — LightGBM's main advantage (training speed at large scale via histogram-based splitting) doesn't materialize on a 7K-row dataset, where XGBoost's exact-greedy splits were both faster to train and comparably accurate. LightGBM remains the better choice at a larger scale (100K+ rows), and is a natural upgrade path if this pipeline is later applied to a bigger customer base.

**Final model: XGBoost Classifier**

| Metric | Value | What it means for the business |
|---|---|---|
| ROC-AUC | **0.838** | Strong separation between customers who churn and who don't |
| Recall @ threshold | **78.9%** | Catches ~4 out of every 5 customers who actually churn |
| Precision @ threshold | **52.3%** | About half of flagged customers genuinely churn — acceptable tradeoff, since the cost of an unnecessary retention call is far lower than the cost of missing a real churner |
| F1 Score | **0.629** | Balanced view of the precision/recall tradeoff |
| Decision threshold | **0.308** (not 0.5) | Tuned deliberately — see "Challenges" below |

**Hyperparameters (via RandomizedSearchCV, 5-fold StratifiedKFold, 20 iterations):**
```
n_estimators: 400
max_depth: 4
learning_rate: 0.05
subsample: 0.9
colsample_bytree: 0.8
min_child_weight: 3
```

**Top churn drivers (SHAP, aggregate):** Month-to-month contract, Fiber optic internet service, short tenure — consistent with the EDA findings.

---

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness check + Groq availability status |
| `/predict` | POST | Single customer — probability, risk tier, SHAP factors, plain-English explanation, revenue at risk |
| `/predict/whatif` | POST | ROI simulator — test a specific retention offer before making it |
| `/predict/batch` | POST | Score up to 100 customers in one call, ranked, with executive summary |

### `POST /predict` — Sample Response
```json
{
  "churn_probability": 0.8114,
  "risk_level": "CRITICAL",
  "top_factors": [
    {
      "feature": "Contract_Month-to-month",
      "impact": "+0.4985",
      "direction": "increases churn risk"
    },
    {
      "feature": "InternetService_Fiber optic",
      "impact": "+0.3041",
      "direction": "increases churn risk"
    },
    {
      "feature": "MultipleLines_Yes",
      "impact": "-0.2839",
      "direction": "decreases churn risk"
    }
  ],
  "business_impact": "Expected revenue at risk: ₹930/year (₹96/month × 12 × 81% churn probability)",
  "recommended_action": "Immediate retention call — offer contract upgrade + 20% discount",
  "explanation": "This customer is at high risk of churning due to their month-to-month contract and fiber optic internet service, which increases the likelihood of them leaving. Their short tenure of just 3 months also suggests they may not be fully invested in our services."
}
```

Note the difference from a typical churn API response: this doesn't just return a number, it returns a *reason* and a *next action*, in the same call.

### `POST /predict/batch` — Sample Response (3 customers)
```json
{
    "predictions": [
        {
            "churn_probability": 0.8114,
            "risk_level": "CRITICAL",
            "top_factors": [
                {
                    "feature": "Contract_Month-to-month",
                    "impact": "+0.4985",
                    "direction": "increases churn risk"
                },
                {
                    "feature": "InternetService_Fiber optic",
                    "impact": "+0.3041",
                    "direction": "increases churn risk"
                },
                {
                    "feature": "MultipleLines_Yes",
                    "impact": "-0.2839",
                    "direction": "decreases churn risk"
                }
            ],
            "business_impact": "Expected revenue at risk: ₹930/year (₹96/month × 12 × 81% churn probability)",
            "recommended_action": "Immediate retention call — offer contract upgrade + 20% discount",
            "explanation": "This customer is at high risk of churning due to their month-to-month contract and fiber optic internet service, which increases the likelihood of them leaving. Their short tenure of just 3 months also suggests they may not be fully invested in our services. The customer's low monthly charges and lack of online security add-on further indicate a potential churn risk."
        },
        {
            "churn_probability": 0.0148,
            "risk_level": "LOW",
            "top_factors": [
                {
                    "feature": "Contract_Month-to-month",
                    "impact": "-0.8584",
                    "direction": "decreases churn risk"
                },
                {
                    "feature": "tenure",
                    "impact": "-0.6549",
                    "direction": "decreases churn risk"
                },
                {
                    "feature": "Contract_Two year",
                    "impact": "-0.5092",
                    "direction": "decreases churn risk"
                }
            ],
            "business_impact": "Expected revenue at risk: ₹8/year (₹45/month × 12 × 1% churn probability)",
            "recommended_action": "No action needed — customer relationship is healthy",
            "explanation": "This customer has a low churn risk due to their long tenure of 60 months with our service. Their two-year contract is also in effect, which reduces their likelihood of switching. Additionally, their monthly charges are relatively low at ₹45.0, indicating they may not be seeking better value elsewhere."
        },
        {
            "churn_probability": 0.7347,
            "risk_level": "HIGH",
            "top_factors": [
                {
                    "feature": "Contract_Month-to-month",
                    "impact": "+0.5973",
                    "direction": "increases churn risk"
                },
                {
                    "feature": "InternetService_Fiber optic",
                    "impact": "+0.3274",
                    "direction": "increases churn risk"
                },
                {
                    "feature": "PaymentMethod_Electronic check",
                    "impact": "+0.2350",
                    "direction": "increases churn risk"
                }
            ],
            "business_impact": "Expected revenue at risk: ₹620/year (₹70/month × 12 × 73% churn probability)",
            "recommended_action": "Flag for retention team — personalised offer within 48 hours",
            "explanation": "This customer is at HIGH risk of churning due to their month-to-month contract, which provides no long-term commitment to our services. Additionally, their fiber optic internet service and payment method via electronic check may indicate a lack of loyalty or flexibility in their payment options."
        }
    ],
    "summary": {
        "total_customers": 3,
        "critical_risk_count": 1,
        "high_risk_count": 1,
        "medium_risk_count": 0,
        "low_risk_count": 1,
        "average_churn_probability": 0.5203,
        "total_annual_revenue_at_risk": 1990.2,
        "top_churn_drivers": [
            "Contract_Month-to-month",
            "InternetService_Fiber optic",
            "tenure"
        ]
    }
}
```
A CRM can call this nightly against the full customer base and get a ranked worklist for the retention team each morning — no manual triage needed.

### `POST /predict/whatif` — The Core Differentiator

Most churn tools stop at prediction. This endpoint answers the question that actually drives revenue decisions: **"If I make this specific change to this specific customer, is it worth it?"**

Example scenario:
> *"If I upgrade this customer from Month-to-month to a Two-year contract AND add Online Security, how much does churn risk drop? What annual revenue am I protecting by doing this?"*

The endpoint takes the original customer profile plus a proposed set of changes, runs the model twice (before/after), and returns the delta in both churn probability and protected revenue — so a retention agent (or an automated CRM workflow) can evaluate an offer's ROI *before* it's made, not after.

This turns the model from a passive scorer into an active decision tool — the kind of feature that separates a hireable AI engineer from someone who followed a Kaggle tutorial.

---

## Challenges & How They Were Solved

Real projects have friction. Documenting it honestly is more credible to a client or interviewer than pretending it was smooth.

**1. Model selection wasn't obvious — XGBoost vs. LightGBM**
LightGBM is the more "modern" choice and is widely adopted in production ML at scale (Microsoft's own recommendation for large tabular datasets), so it was tested seriously as the final candidate. In practice, on this dataset's size (~7K rows), LightGBM's leaf-wise growth didn't outperform XGBoost's depth-wise growth on ROC-AUC, and XGBoost trained faster besides. Decision: use XGBoost for this dataset size, but treat LightGBM as the documented upgrade path if this pipeline is ever applied to a larger customer base (100K+ rows), where its histogram-based splitting starts to pay off.

**2. Categorical noise: "No internet service" and "No phone service"  vs. "No"**
Several service columns (`OnlineSecurity`, `TechSupport`, `StreamingTV`, etc.) had three category values instead of two: `Yes`, `No`, and `No internet service`. Left as-is, one-hot encoding these created redundant, correlated columns — "No internet service" is really just a specific case of "No" for that feature, and having it as a separate category diluted the signal the model could learn from the true Yes/No split. Collapsing `No internet service` → `No` reduced noise and measurably cleaned up feature importance rankings.

**3. Data leakage risk in the What-If Simulator**
The first version of the what-if logic mutated the original customer dictionary in place when applying the "proposed changes" for simulation — which meant the "before" state was silently overwritten before the "before" prediction was made, or was contaminated for any later use of that same customer object. Fixed by using `customer.copy()` (a defensive shallow copy) before applying simulated changes, so the original profile stays intact for the baseline prediction and the modified version is fully isolated. This is a subtle bug that's easy to miss and easy to explain in an interview — it demonstrates actual engineering care, not just model-fitting.

**4. API key / provider mix-up**
Initially configured for `XAI_API_KEY` / `grok-3-mini`, but the actual provider integrated was Groq (`GROQ_API_KEY` / `llama3-8b-8192`). Caught during endpoint testing — a reminder to verify provider/model names explicitly rather than assuming from naming similarity.

---

## Business Impact

Applied across a customer base at this dataset's churn rate (~26.5%) and average monthly revenue per customer, the model's recall (78.9%) means the retention team can act on roughly 4 out of every 5 customers who would otherwise leave — before they leave. Framed the way this project frames every prediction: not "84% ROC-AUC," but **rupees protected per month**, because that's the number that gets a retention team's budget approved.

---

## Tech Stack

Python · FastAPI · XGBoost · scikit-learn · SHAP · SMOTE (imbalanced-learn) · Groq API · Render

---

## Known Limitations

- Free-tier hosting (Render) spins down after ~15 minutes of inactivity; first request after idle takes 30–50 seconds to wake up.
- Trained on Telco-industry data specifically — the feature set (contract type, internet service, streaming add-ons) would need adapting for other verticals (SaaS, e-commerce subscriptions).
- Precision of 52.3% means roughly half of flagged "at-risk" customers won't actually churn — an intentional tradeoff favoring recall, but worth knowing before budgeting retention offers at scale.
- LightGBM was evaluated but not deployed; revisit if this pipeline is scaled to a significantly larger customer base.

---

## Run Locally

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

API docs (interactive): `http://localhost:8000/docs`

## Live Demo

API: `https://customer-churn-predictor-hju5.onrender.com/docs`

GitHub: `https://github.com/OmsinghThakur09/Customer-churn-predictor-api`

---