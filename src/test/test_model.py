import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the core functions from the main pipeline script
# Note: For running tests, you usually need to install the project
# or ensure the source directory is in the PYTHONPATH.
# Assuming standard test runner setup (e.g., running `pytest` from the project root)
from src.model_pipeline import (
    calculate_rfms,
    extract_temporal_features,
    create_target_variable,
    evaluate_model,
    load_and_simulate_raw_data
)

# --- Fixtures for Testing ---

@pytest.fixture
def sample_raw_data():
    """Provides a small, controlled raw data sample for deterministic testing."""
    start_time = datetime(2025, 1, 1)
    
    data = {
        'CustomerID': ['CUST_A', 'CUST_A', 'CUST_B', 'CUST_C', 'CUST_C', 'CUST_C', 'CUST_D'],
        'TransactionAmount': [100.0, 50.0, 200.0, 10.0, 15.0, 5.0, 500.0],
        'TransactionStartTime': [
            start_time,                                # A: Day 0
            start_time + timedelta(days=5),            # A: Day 5 (Recency: 6 days)
            start_time + timedelta(days=20),           # B: Day 20 (Recency: 1 day)
            start_time + timedelta(days=10),           # C: Day 10
            start_time + timedelta(days=15),           # C: Day 15
            start_time + timedelta(days=20),           # C: Day 20 (Recency: 1 day)
            start_time + timedelta(days=1),            # D: Day 1 (Recency: 20 days)
        ]
    }
    df = pd.DataFrame(data)
    # The current snapshot date used in calculate_rfms is max(time) + 1 day.
    # Here, snapshot_date = Day 21 (datetime(2025, 1, 22))
    return df

@pytest.fixture
def target_creation_data():
    """Provides a larger, randomized dataset to test clustering functionality."""
    return load_and_simulate_raw_data(num_samples=100)

# --- Test RFMS Calculation ---

def test_calculate_rfms_columns(sample_raw_data):
    """Checks if calculate_rfms returns the expected column names."""
    df_rfms = calculate_rfms(sample_raw_data)
    expected_cols = ['CustomerID', 'Recency_Days', 'Frequency', 'Monetary_Value', 'Std_Deviation']
    assert all(col in df_rfms.columns for col in expected_cols)
    assert df_rfms.shape[0] == 4 # 4 unique customers

def test_calculate_rfms_values(sample_raw_data):
    """Checks specific calculated RFMS values for accuracy."""
    df_rfms = calculate_rfms(sample_raw_data)
    
    # Customer A: Last transaction Day 5. Recency = Day 21 - Day 5 = 16 days. Freq=2. Mon=(100+50)/2 = 75
    cust_a = df_rfms[df_rfms['CustomerID'] == 'CUST_A'].iloc[0]
    assert cust_a['Recency_Days'] == 16
    assert cust_a['Frequency'] == 2
    assert cust_a['Monetary_Value'] == 75.0
    
    # Customer D: Only one transaction. Std Dev should be 0 (due to fillna(0) logic).
    cust_d = df_rfms[df_rfms['CustomerID'] == 'CUST_D'].iloc[0]
    assert cust_d['Std_Deviation'] == 0.0

# --- Test Temporal Feature Extraction ---

def test_extract_temporal_features_columns(sample_raw_data):
    """Checks if temporal feature extraction returns the correct columns."""
    df_temp = extract_temporal_features(sample_raw_data)
    expected_cols = ['CustomerID', 'TransactionHour', 'TransactionMonth', 'TransactionDayOfWeek']
    assert all(col in df_temp.columns for col in expected_cols)
    assert df_temp.shape[0] == 4

def test_extract_temporal_features_values(sample_raw_data):
    """Checks specific temporal feature values (based on last transaction time)."""
    df_temp = extract_temporal_features(sample_raw_data)
    
    # Last transaction for A, B, C, D all happened at 00:00:00 (hour 0)
    assert (df_temp['TransactionHour'] == 0).all()
    # All transactions happened in January (month 1)
    assert (df_temp['TransactionMonth'] == 1).all()

# --- Test Target Creation (Clustering) ---

def test_create_target_variable_output(target_creation_data):
    """Checks if target creation produces the correct structure and type."""
    df_target = create_target_variable(target_creation_data)
    assert df_target.shape[0] > 0
    assert 'CustomerID' in df_target.columns
    assert 'is_high_risk' in df_target.columns
    # Target column must be binary
    assert set(df_target['is_high_risk'].unique()).issubset({0, 1})

# --- Test Evaluation Metrics ---

def test_evaluate_model_perfect_score():
    """Checks evaluation metrics for a perfect model."""
    y_true = np.array([0, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.2, 0.9, 0.9])
    y_pred = np.array([0, 0, 1, 1])
    
    metrics = evaluate_model(y_true, y_pred_proba, y_pred)
    
    # Perfect score means all discrimination metrics should be 1.0
    assert metrics['AUC'] == 1.0
    assert metrics['GINI'] == 1.0
    assert metrics['KS_Statistic'] == 1.0
    assert metrics['Accuracy'] == 1.0

def test_evaluate_model_random_score():
    """Checks AUC/GINI for a model performing at random (0.5 AUC, 0 GINI)."""
    y_true = np.array([0, 0, 1, 1])
    y_pred_proba = np.array([0.5, 0.5, 0.5, 0.5]) # Cannot differentiate
    y_pred = np.array([0, 0, 0, 0]) # Assuming threshold is 0.5

    # We must handle the edge case where all probas are the same, ROC AUC might fail.
    # Let's ensure probas are slightly varied but still random performance
    y_pred_proba_random = np.array([0.4, 0.6, 0.4, 0.6])
    y_pred_random = np.array([0, 1, 0, 1]) # Random predictions based on threshold
    
    metrics = evaluate_model(y_true, y_pred_proba_random, y_pred_random)

    # AUC should be close to 0.5 for random assignment
    assert np.isclose(metrics['AUC'], 0.5, atol=0.01)
    # GINI should be close to 0 for random assignment
    assert np.isclose(metrics['GINI'], 0.0, atol=0.02)