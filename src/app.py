import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# --- Pydantic Schemas for API ---

class Transaction(BaseModel):
    """Schema for a single transaction record."""
    CustomerID: str
    TransactionAmount: float = Field(..., gt=0)
    TransactionStartTime: datetime

class CustomerData(BaseModel):
    """Schema for the input request: a list of transactions for a single customer."""
    transactions: List[Transaction]

class PredictionResponse(BaseModel):
    """Schema for the prediction output."""
    CustomerID: str
    Risk_Probability: float = Field(..., ge=0, le=1)
    Risk_Level: str

# --- Model Loading and Initialization ---

# Initialize FastAPI app
app = FastAPI(
    title="Alternative Data Credit Risk API",
    description="Predicts credit risk probability based on customer's historical transactions.",
    version="1.0.0"
)

MODEL_PATH = 'models/best_model_pipeline.pkl'
model_pipeline = None

@app.on_event("startup")
def load_model():
    """Load the trained model pipeline on startup."""
    global model_pipeline
    try:
        model_pipeline = joblib.load(MODEL_PATH)
        print(f"Model successfully loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {MODEL_PATH}. Run src/model_pipeline.py first!")
        raise HTTPException(
            status_code=500, 
            detail="Model artifact not found. Please train the model first."
        )
    except Exception as e:
        print(f"ERROR loading model: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to load model: {e}"
        )

# --- Feature Engineering Functions (copied for inference completeness) ---

def calculate_rfms_inference(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Calculates Recency, Frequency, Monetary Value, and Std Deviation (RFMS)."""
    # Use the latest transaction time in the input data as the snapshot date
    snapshot_date = df_raw['TransactionStartTime'].max() + pd.Timedelta(days=1)
    
    rfm_df = df_raw.groupby('CustomerID').agg(
        Recency_Days=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
        Frequency=('CustomerID', 'count'),
        Monetary_Value=('TransactionAmount', 'mean'),
        Std_Deviation=('TransactionAmount', 'std')
    ).fillna(0).reset_index()
    
    return rfm_df.drop(columns=['CustomerID']) # Drop ID as it's not a feature

def extract_temporal_features_inference(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Extracts explicit temporal features based on the *last* transaction."""
    # Find the last transaction time for the single customer
    last_transaction_time = df_raw['TransactionStartTime'].max()
    
    data = {
        'TransactionHour': [last_transaction_time.hour],
        'TransactionMonth': [last_transaction_time.month],
        'TransactionDayOfWeek': [last_transaction_time.dayofweek]
    }
    return pd.DataFrame(data)

def preprocess_input(transactions: List[Transaction]) -> pd.DataFrame:
    """Converts raw transaction list to the single-row feature DataFrame."""
    
    # Convert Pydantic list to Pandas DataFrame
    raw_data = pd.DataFrame([t.model_dump() for t in transactions])
    
    # 1. Calculate RFMS features
    rfms_features = calculate_rfms_inference(raw_data)
    
    # 2. Extract Temporal features
    temporal_features = extract_temporal_features_inference(raw_data)
    
    # 3. Combine into the final input row (must maintain column order used during training!)
    # The order expected by the pipeline is: RFMS features + Temporal features
    final_features = pd.concat([rfms_features, temporal_features], axis=1)
    
    # Check for NaN/Inf values that could break the prediction
    if final_features.isnull().values.any() or np.isinf(final_features.values).any():
        raise HTTPException(status_code=400, detail="Generated features contain invalid (NaN/Inf) values.")

    return final_features

# --- API Endpoints ---

@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "Credit Risk Prediction Service is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict_risk(data: CustomerData):
    """
    Accepts a list of a customer's transactions and returns the calculated credit risk probability.
    """
    if not model_pipeline:
        raise HTTPException(status_code=503, detail="Model is not loaded. Service unavailable.")

    # Validation: Ensure all transactions belong to the same customer
    customer_ids = {t.CustomerID for t in data.transactions}
    if len(customer_ids) != 1:
        raise HTTPException(status_code=400, detail="Input must contain transactions for exactly one CustomerID.")
    
    customer_id = next(iter(customer_ids))
    
    try:
        # 1. Feature Engineering
        input_df = preprocess_input(data.transactions)
        
        # 2. Prediction
        # Predict probability of class 1 (high risk)
        probability_high_risk = model_pipeline.predict_proba(input_df)[0, 1]
        
        # 3. Determine Risk Level (simple threshold for interpretation)
        threshold = 0.5  # Example threshold
        risk_level = "High Risk" if probability_high_risk >= threshold else "Low Risk"

        return PredictionResponse(
            CustomerID=customer_id,
            Risk_Probability=probability_high_risk,
            Risk_Level=risk_level
        )

    except HTTPException:
        # Re-raise HTTP exceptions (e.g., from preprocess_input)
        raise
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {e}")