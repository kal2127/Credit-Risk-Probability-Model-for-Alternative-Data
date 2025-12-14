import pandas as pd
import numpy as np
from pathlib import Path

# --- Configuration (using robust path resolution) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FILE_NAME = 'data.csv' 
FILE_PATH = PROJECT_ROOT / 'data' / FILE_NAME 

# Set a reference date for Recency calculation. 
# We explicitly remove any time component to ensure it is tz-naive.
RECENCY_DATE = pd.to_datetime('2025-01-01').normalize() 


def load_data(file_path):
    """Loads the transactional data and performs initial cleaning."""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully with {len(df)} records.")
    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}. Please check the path and file name.")
        return None

    # --- CRITICAL COLUMN NAME MAPPING ---
    try:
        df = df.rename(columns={
            'CustomerId': 'Customer_ID',      
            'TransactionStartTime': 'Transaction_Date', 
            'Amount': 'Amount'                
        })
    except KeyError as e:
        print(f"\nFATAL ERROR: Column name mismatch! Failed to find column: {e}")
        print("Please verify your 'CustomerId', 'TransactionStartTime', and 'Amount' column names.")
        return None

    # Convert to datetime object
    df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])

    # *** FIX: Remove Time Zone (TZ) Information to resolve the TypeError ***
    if df['Transaction_Date'].dt.tz is not None:
        print("Time zone information detected and removed from transaction dates.")
        df['Transaction_Date'] = df['Transaction_Date'].dt.tz_localize(None)
    
    return df[['Customer_ID', 'Transaction_Date', 'Amount']]


def calculate_rfms(df, recency_ref_date):
    """Calculates Recency, Frequency, Monetary Value, and Standard Deviation (RFMS) features."""
    
    # 1. R: Recency (Days since last transaction)
    recency_df = df.groupby('Customer_ID')['Transaction_Date'].max().reset_index()
    # This calculation is now safe because both dates are tz-naive.
    recency_df['Recency'] = (recency_ref_date - recency_df['Transaction_Date']).dt.days

    # 2. F: Frequency (Total number of transactions)
    frequency_df = df.groupby('Customer_ID').size().reset_index(name='Frequency')

    # 3. M: Monetary Value (Average transaction amount)
    monetary_df = df.groupby('Customer_ID')['Amount'].mean().reset_index(name='Monetary_Value')

    # 4. S: Standard Deviation (Volatility of transaction amount - a proxy for risk)
    std_dev_df = df.groupby('Customer_ID')['Amount'].std().fillna(0).reset_index(name='Std_Deviation')

    # Merge all RFMS features
    rfms_df = recency_df[['Customer_ID', 'Recency']].merge(
        frequency_df, on='Customer_ID'
    ).merge(
        monetary_df, on='Customer_ID'
    ).merge(
        std_dev_df, on='Customer_ID'
    )
    
    return rfms_df


# In feature_engineering.py, replace the define_proxy function:

def define_proxy(rfms_df):
    """
    Creates a numerical RFMS Score and defines the binary Default_Proxy variable (target).
    
    NOTE: Using 4 bins (quartiles) instead of 5 (quintiles) to handle data sparsity 
    (many zeros in Std_Deviation), which caused the ValueError.
    """
    NUM_BINS = 4
    
    # R Score: Lower Recency (fewer days) is better (score 4)
    rfms_df['R_Score'] = pd.qcut(rfms_df['Recency'], NUM_BINS, labels=[4, 3, 2, 1], duplicates='drop').astype(int) 
    
    # F Score: Higher Frequency is better (score 4)
    rfms_df['F_Score'] = pd.qcut(rfms_df['Frequency'], NUM_BINS, labels=[1, 2, 3, 4], duplicates='drop').astype(int)
    
    # M Score: Higher Monetary Value is better (score 4)
    rfms_df['M_Score'] = pd.qcut(rfms_df['Monetary_Value'], NUM_BINS, labels=[1, 2, 3, 4], duplicates='drop').astype(int)
    
    # S Score: Lower Std Deviation is better (score 4)
    rfms_df['S_Score'] = pd.qcut(rfms_df['Std_Deviation'], NUM_BINS, labels=[4, 3, 2, 1], duplicates='drop').astype(int) 

    # Combined RFMS Score (Max score is now 4*4 = 16)
    rfms_df['RFMS_Score'] = rfms_df['R_Score'] + rfms_df['F_Score'] + rfms_df['M_Score'] + rfms_df['S_Score']

    # Define the Proxy Threshold (20th percentile of the combined score)
    BAD_CUSTOMER_THRESHOLD = rfms_df['RFMS_Score'].quantile(0.20)

    # Default_Proxy: 1 if RFMS_Score is below or at the threshold (Bad), 0 otherwise (Good)
    rfms_df['Default_Proxy'] = np.where(rfms_df['RFMS_Score'] <= BAD_CUSTOMER_THRESHOLD, 1, 0)
    
    return rfms_df


def main():
    """Main function to run the feature engineering pipeline."""
    
    # 1. Load and clean data
    transaction_df = load_data(FILE_PATH)
    if transaction_df is None:
        return

    # 2. Calculate RFMS features
    rfms_features = calculate_rfms(transaction_df, RECENCY_DATE)

    # 3. Define the Proxy target variable
    final_df = define_proxy(rfms_features)

    # 4. Output Summary
    print("\n--- RFMS Feature & Proxy Summary ---")
    print(final_df[['Recency', 'Frequency', 'Monetary_Value', 'Std_Deviation', 'RFMS_Score', 'Default_Proxy']].describe())
    print(f"\nProportion of 'Bad' customers (Default_Proxy=1): {final_df['Default_Proxy'].mean():.2%}")

    # 5. Save the final dataset
    final_df.to_csv(PROJECT_ROOT / 'data' / 'rfms_features_with_proxy.csv', index=False)
    print(f"\nRFMS features and the Default Proxy saved to {PROJECT_ROOT / 'data' / 'rfms_features_with_proxy.csv'}")

if __name__ == "__main__":
    main()