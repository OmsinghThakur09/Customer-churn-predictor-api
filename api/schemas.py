# api/schemas.py

from pydantic import BaseModel, Field
from typing import List, Dict, Any

"""
BaseModel is the parent class for all request/response shapes. 
Field lets you add metadata like descriptions and example values that show up in Swagger docs.
The Config class with json_schema_extra populates the "Example Value" in Swagger UI automatically.
Clients see exactly what JSON to send without reading any docs."""


# CustomerInput(expected format and datatypes of customer input)
class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class ShapFactor(BaseModel):
    feature: str
    impact: str
    direction: str


class PredictionResponse(BaseModel):
    churn_probability: float
    risk_level: str
    top_factors: List[ShapFactor]
    business_impact: str
    recommended_action: str
    explanation: str


class WhatIfInput(BaseModel):
    customer: CustomerInput
    proposed_change: Dict[str, Any] = Field(
        ...,
        description="Feature Key-value pairs to change",
        json_schema_extra={
            "example": {"Contract": "Two year", "OnlineSecurity": "yes"}
        }
    )


class WhatIfResponse(BaseModel):
    current: PredictionResponse
    proposed: PredictionResponse
    risk_reduction_pct: float
    intervention_value: str
    recommendation: str
