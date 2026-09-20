"""Shared, serializable transformations used by the notebook and FastAPI."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

YES_NO = ["Yes", "No"]
ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
          "StreamingTV", "StreamingMovies"]
CATEGORIES = {
    "gender": ["Female", "Male"], "Partner": YES_NO, "Dependents": YES_NO,
    "PhoneService": YES_NO, "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    **{c: ["Yes", "No", "No internet service"] for c in ADDONS},
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod": ["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"],
}
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
RAW_FEATURES = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
                "PhoneService", "MultipleLines", "InternetService", *ADDONS,
                "Contract", "PaperlessBilling", "PaymentMethod",
                "MonthlyCharges", "TotalCharges"]


def clean_features(frame):
    """Stateless schema normalization: no statistics are learned here."""
    missing = sorted(set(RAW_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    result = frame.loc[:, RAW_FEATURES].copy()
    for col in CATEGORIES:
        values = result[col].astype("string").str.strip().replace("", pd.NA)
        canonical = {value.casefold(): value for value in CATEGORIES[col]}
        mapped = values.str.casefold().map(canonical)
        invalid = values.notna() & mapped.isna()
        if invalid.any():
            raise ValueError(f"Invalid {col} values: {values[invalid].unique().tolist()}")
        result[col] = mapped.astype(object).where(mapped.notna(), np.nan)
    for col in NUMERIC:
        result[col] = pd.to_numeric(result[col], errors="coerce").astype(float)
        result[col] = result[col].replace([np.inf, -np.inf], np.nan)
        if (result[col].dropna() < 0).any():
            raise ValueError(f"{col} contains negative values; correct them before training.")
    if not result["SeniorCitizen"].dropna().isin([0, 1]).all():
        raise ValueError("SeniorCitizen must be 0 or 1 (or missing).")
    if (result["tenure"].dropna() % 1 != 0).any():
        raise ValueError("tenure must contain whole numbers of months (or missing).")
    return result


class TelcoFeatures(BaseEstimator, TransformerMixin):
    """Cleaning and row-wise features; safe inside cross-validation and pickle."""
    def fit(self, X, y=None):
        clean_features(X)
        return self

    def transform(self, X):
        result = clean_features(X)
        result["TenureGroup"] = pd.cut(
            result["tenure"], bins=[-np.inf, 12, 24, 48, np.inf],
            labels=["0-12 months", "13-24 months", "25-48 months", "49+ months"],
        ).astype(object)
        # Eight services: phone, internet, and six internet add-ons.
        # A count with any unknown component remains unknown, then is imputed.
        service_cols = ["PhoneService", "InternetService", *ADDONS]
        count = result[["PhoneService", *ADDONS]].eq("Yes").sum(axis=1).astype(float)
        count += result["InternetService"].isin(["DSL", "Fiber optic"]).astype(float)
        result["ServiceCount"] = count.mask(result[service_cols].isna().any(axis=1))
        return result


def make_pipeline(**tree_params):
    numeric = NUMERIC + ["ServiceCount"]
    categorical = list(CATEGORIES) + ["TenureGroup"]
    preprocess = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ], remainder="drop", verbose_feature_names_out=True)
    return Pipeline([
        ("features", TelcoFeatures()),
        ("preprocess", preprocess),
        ("model", DecisionTreeClassifier(random_state=42, **tree_params)),
    ])


def prediction_result(model, customer):
    frame = pd.DataFrame([customer])
    label = int(model.predict(frame)[0])
    index = list(model.classes_).index(1)
    probability = float(model.predict_proba(frame)[0, index])
    return {"prediction": "Yes" if label == 1 else "No",
            "churn_probability": round(probability, 6)}
