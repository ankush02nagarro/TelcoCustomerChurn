"""Run from project root: python -m uvicorn app:app --reload"""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from preprocessing import prediction_result

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("CHURN_MODEL_PATH", ROOT / "model/churn_model.pkl"))
YesNo = Literal["Yes", "No"]
Addon = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    """All predictor keys are required. Null values use training-time imputation."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)
    customerID: str | None = None
    gender: Literal["Female", "Male"] | None
    SeniorCitizen: Literal[0, 1] | None
    Partner: YesNo | None
    Dependents: YesNo | None
    tenure: int | None = Field(ge=0)
    PhoneService: YesNo | None
    MultipleLines: Literal["Yes", "No", "No phone service"] | None
    InternetService: Literal["DSL", "Fiber optic", "No"] | None
    OnlineSecurity: Addon | None
    OnlineBackup: Addon | None
    DeviceProtection: Addon | None
    TechSupport: Addon | None
    StreamingTV: Addon | None
    StreamingMovies: Addon | None
    Contract: Literal["Month-to-month", "One year", "Two year"] | None
    PaperlessBilling: YesNo | None
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)",
                           "Credit card (automatic)"] | None
    MonthlyCharges: float | None = Field(ge=0)
    TotalCharges: float | None = Field(ge=0)

    @field_validator("TotalCharges", mode="before")
    @classmethod
    def blank_total_is_missing(cls, value):
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", mode="before")
    @classmethod
    def reject_boolean_numbers(cls, value):
        if isinstance(value, bool):
            raise ValueError("Use a number, not a boolean.")
        return value


class Prediction(BaseModel):
    prediction: Literal["Yes", "No"]
    churn_probability: float = Field(ge=0, le=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run the notebook first.")
    app.state.model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(title="Telco Customer Churn API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer, request: Request):
    try:
        return prediction_result(request.app.state.model, customer.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
