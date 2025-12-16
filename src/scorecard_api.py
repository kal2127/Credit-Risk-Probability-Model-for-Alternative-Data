from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import math

# --- Configuration ---
COEFF_PATH = 'models/logreg_coefficients.csv'
# NOTE: The WoE lookup table is critical for a complete API pipeline (raw data -> WoE -> Score).
# We assume the raw WoE features are passed directly to this endpoint for simplicity here.
BASE_ODDS_SCORE = 600
PDO = 20 # Points to Double the Odds (standard for credit scorecards, typically 20 or 50)

# --- Data Structures for Scoring ---

class WoEFeature(BaseModel):
    """Schema for the inputs that have already been WoE-transformed."""
    Recency_WoE: float
    Frequency_WoE: float
    Monetary_Value_WoE: float
    Std_Deviation_WoE: float

# --- FastAPI Initialization and Coefficient Loading ---

app = FastAPI(
    title="Bati Bank Credit Scorecard API",
    description="Serves the regulatory-compliant Logistic Regression Scorecard using WoE-transformed features."
)

coefficients = {}

def load_coefficients():
    """Load the pre-calculated Logistic Regression coefficients."""
    global coefficients
    try:
        coeff_df = pd.read_csv(COEFF_PATH)
        # Store coefficients in a dictionary for easy lookup
        for _, row in coeff_df.iterrows():
            coefficients[row['Feature']] = row['Coefficient']

        # Ensure all features are present
        required_features = ['Intercept', 'Recency_WoE', 'Frequency_WoE', 'Monetary_Value_WoE', 'Std_Deviation_WoE']
        if not all(feature in coefficients for feature in required_features):
            raise ValueError("Missing required coefficients in the CSV file.")

        print(f"Loaded coefficients for {len(coefficients)} features.")
    except FileNotFoundError:
        print(f"Error: Coefficient file not found at {COEFF_PATH}. Run model_building.py first.")
        # Re-raise the exception to fail startup if critical model files are missing
        raise HTTPException(status_code=500, detail="Model coefficients not found.")
    except Exception as e:
        print(f"Error loading coefficients: {e}")
        raise HTTPException(status_code=500, detail="Failed to load model coefficients.")

# Execute coefficient loading when the app starts
load_coefficients()


# --- Scoring Logic ---

def calculate_score(woe_features: WoEFeature):
    """
    Calculates the final scorecard points based on the Logistic Regression formula.
    Score = Offset - (Factor * Logit)
    """
    
    # 1. Calculate Log-Odds (Logit)
    # Logit = Intercept + sum(Coefficient * WoE)
    
    logit = coefficients.get('Intercept', 0.0)
    
    # Sum of (Coefficient * WoE)
    logit += coefficients.get('Recency_WoE', 0.0) * woe_features.Recency_WoE
    logit += coefficients.get('Frequency_WoE', 0.0) * woe_features.Frequency_WoE
    logit += coefficients.get('Monetary_Value_WoE', 0.0) * woe_features.Monetary_Value_WoE
    logit += coefficients.get('Std_Deviation_WoE', 0.0) * woe_features.Std_Deviation_WoE

    # 2. Scorecard Transformation
    
    # Factor = PDO / ln(2)
    factor = PDO / math.log(2)
    
    # Offset (Adjusted Base Score)
    # The intercept is the base log-odds. We map the BASE_ODDS_SCORE to this log-odds.
    offset = BASE_ODDS_SCORE - (factor * coefficients.get('Intercept', 0.0))
    
    # Final Score Calculation
    final_score = offset - (factor * logit)

    return int(round(final_score))


# --- API Endpoint ---

@app.post("/scorecard/predict", response_model=dict)
async def predict_score(data: WoEFeature):
    """
    Accepts pre-transformed WoE features and returns the final credit score.
    
    To run this service locally, you would typically use:
    uvicorn src.scorecard_api:app --reload
    
    Example POST body (replace with actual WoE values):
    {
      "Recency_WoE": 1.5, 
      "Frequency_WoE": 0.8, 
      "Monetary_Value_WoE": 0.2, 
      "Std_Deviation_WoE": -0.1
    }
    """
    
    try:
        score = calculate_score(data)
        
        return {
            "credit_score": score,
            "interpretation": f"Score {score} is derived from the highly interpretable linear scorecard model.",
            "metrics": {
                "base_score": BASE_ODDS_SCORE,
                "pdo": PDO,
                "model_status": "Ready for Production"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal scoring error: {e}")