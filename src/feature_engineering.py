import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
from typing import Tuple

# --- 1. SETUP LOGGING ---
# Define a standard logger for pipeline monitoring
LOG_FILE = 'feature_engineering.log'
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(LOG_FILE, mode='w'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# --- 2. CONFIGURATION ---
DATE_FORMAT = '%Y-%m-%dT%H:%M:%S'
REFERENCE_DATE = datetime(2025, 1, 1)
DATA_PATH = '../data/data.csv'
OUTPUT_PATH = '../data/rfms_features_with_proxy.csv'
PROXY_QUANTILE = 0.20 # Define the bottom X% of RFMS score as the Default Proxy (1)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the raw transactional data, standardizes column names, and converts types.

    Args:
        file_path (str): The path to the raw CSV data file.

    Returns:
        pd.DataFrame: The cleaned DataFrame ready for aggregation.
    """
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        # Standardize column names
        df.rename(columns={
            'CustomerId': 'Customer_ID', 
            'TransactionStartTime': 'Transaction_Date'
        }, inplace=True)
        # Convert date column
        df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], format=DATE_FORMAT)
        logger.info(f"Data loaded successfully. Total transactions: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Error loading or cleaning data: {e}")
        raise


def calculate_rfms(df: pd.DataFrame, reference_date: datetime) -> pd.DataFrame:
    """
    Calculates Recency, Frequency, Monetary Value, and Std Deviation (RFMS) features 
    at the customer level.

    Args:
        df (pd.DataFrame): The transactional data.
        reference_date (datetime): The date used to calculate Recency (e.g., today's date or 
                                    a future cut-off date).

    Returns:
        pd.DataFrame: DataFrame containing customer-level RFMS features.
    """
    logger.info("Starting RFMS feature aggregation...")
    
    rfms_df = df.groupby('Customer_ID').agg(
        # Recency: Days since last transaction
        Recency=('Transaction_Date', lambda x: (reference_date - x.max()).days),
        
        # Frequency: Total count of transactions
        Frequency=('Transaction_Date', 'count'),
        
        # Monetary Value: Average transaction amount
        Monetary_Value=('Amount', 'mean'),
        
        # Std Deviation: Volatility of transaction amount
        Std_Deviation=('Amount', 'std')
    )
    
    # Handle NaN values in Std_Deviation for customers with only 1 transaction (Std is undefined)
    # Impute with 0, as zero deviation means stable/no volatility, which is low risk.
    rfms_df['Std_Deviation'] = rfms_df['Std_Deviation'].fillna(0)
    
    logger.info(f"RFMS calculation complete. Total unique customers: {len(rfms_df)}")
    return rfms_df.reset_index()


def define_default_proxy(rfms_df: pd.DataFrame, quantile_threshold: float) -> pd.DataFrame:
    """
    Defines the binary 'Default_Proxy' target variable based on the combined RFMS score.
    Customers in the bottom X% of the RFMS score are flagged as risky (Default_Proxy=1).

    Args:
        rfms_df (pd.DataFrame): DataFrame containing RFMS features.
        quantile_threshold (float): The threshold (e.g., 0.20 for bottom 20%) to define the proxy.

    Returns:
        pd.DataFrame: The DataFrame with the added 'Default_Proxy' column.
    """
    logger.info("Defining the Default Proxy variable based on RFMS score.")
    
    # Simple score: Sum of z-scores (Standardization is required for balanced weighting)
    # Note: Higher Recency is BAD, so we subtract it. Higher F/M/S are generally GOOD.
    z_scores = rfms_df[['Recency', 'Frequency', 'Monetary_Value', 'Std_Deviation']].apply(
        lambda x: (x - x.mean()) / x.std()
    )
    
    # Calculate a combined RFMS Score (higher score = better customer profile)
    rfms_df['RFMS_Score'] = z_scores['Frequency'] + z_scores['Monetary_Value'] + z_scores['Std_Deviation'] - z_scores['Recency']
    
    # Determine the threshold for the bottom X% (risky customers)
    score_threshold = rfms_df['RFMS_Score'].quantile(quantile_threshold)
    
    # Define Default_Proxy: 1 if score is below threshold (risky), 0 otherwise (safe)
    rfms_df['Default_Proxy'] = np.where(
        rfms_df['RFMS_Score'] <= score_threshold, 
        1,  # Bad (Risky)
        0   # Good (Safe)
    )
    
    proxy_count = rfms_df['Default_Proxy'].sum()
    logger.info(f"Proxy defined. Threshold: {score_threshold:.3f}. Bad customers (1): {proxy_count} ({proxy_count/len(rfms_df)*100:.2f}%)")
    
    # Drop the intermediate 'RFMS_Score'
    return rfms_df.drop(columns=['RFMS_Score'])


def save_data(df: pd.DataFrame, file_path: str):
    """
    Saves the final feature DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_path (str): The destination path for the CSV file.
    """
    logger.info(f"Saving final features and proxy to {file_path}")
    df.to_csv(file_path, index=False)
    logger.info("Feature engineering script finished successfully.")


# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        # A. Load and clean the data
        raw_df = load_data(DATA_PATH)
        
        # B. Calculate RFMS features
        rfms_features = calculate_rfms(raw_df, REFERENCE_DATE)
        
        # C. Define the Default Proxy target
        final_df = define_default_proxy(rfms_features, PROXY_QUANTILE)
        
        # D. Save the result
        save_data(final_df, OUTPUT_PATH)
        
    except Exception as e:
        logger.critical(f"Pipeline failed during feature engineering: {e}")