import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Tuple

# --- 1. SETUP LOGGING ---
LOG_FILE = 'woe_iv_transformation.log'
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(LOG_FILE, mode='w'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# --- 2. CONFIGURATION ---
INPUT_PATH = '../data/rfms_features_with_proxy.csv'
OUTPUT_DATA_PATH = '../data/rfms_woe_transformed_data.csv'
LOOKUP_TABLE_PATH = '../models/woe_lookup_table.csv'
TARGET_COLUMN = 'Default_Proxy'

# Features to be transformed (excluding Customer_ID and the target)
FEATURES = ['Recency', 'Frequency', 'Monetary_Value', 'Std_Deviation']


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the RFMS feature data with the defined target proxy.

    Args:
        file_path (str): The path to the RFMS features CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Total customers: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def calculate_woe_iv(data: pd.DataFrame, feature: str, target: str) -> Tuple[pd.DataFrame, float]:
    """
    Calculates Weight of Evidence (WoE) and Information Value (IV) for a feature.

    It uses quartiles (qcut=4) for continuous feature binning.

    Args:
        data (pd.DataFrame): The DataFrame containing the feature and target.
        feature (str): The name of the feature column.
        target (str): The name of the binary target column.

    Returns:
        Tuple[pd.DataFrame, float]: WoE table and the calculated Information Value.
    """
    logger.info(f"Calculating WoE/IV for feature: {feature}")
    
    # 1. Binning the continuous variable using quartiles (q=4)
    # Using 'try/except' to handle errors if a feature has too few unique values for qcut
    try:
        data['Bin'] = pd.qcut(data[feature], q=4, duplicates='drop', precision=0)
    except Exception as e:
        logger.warning(f"Failed to use qcut=4 for {feature}. Using unique values/less bins. Error: {e}")
        data['Bin'] = pd.cut(data[feature], bins=10, duplicates='drop', precision=0) # Fallback binning

    # 2. Group by bins and calculate counts
    woe_table = data.groupby('Bin')[target].agg(
        Total_Count='count',
        Bad_Count='sum',
        Good_Count=lambda x: (x == 0).sum()
    ).reset_index()

    # 3. Calculate Distribution of Good and Bad
    total_good = woe_table['Good_Count'].sum()
    total_bad = woe_table['Bad_Count'].sum()
    
    if total_good == 0 or total_bad == 0:
        logger.warning(f"Total Good or Bad count is zero. WoE/IV calculation skipped for {feature}.")
        return pd.DataFrame(), 0.0

    woe_table['Dist_Good'] = woe_table['Good_Count'] / total_good
    woe_table['Dist_Bad'] = woe_table['Bad_Count'] / total_bad
    
    # Handle log(0) issue by adding a tiny epsilon
    epsilon = 0.00001
    woe_table['Dist_Good'] = woe_table['Dist_Good'].apply(lambda x: max(x, epsilon))
    woe_table['Dist_Bad'] = woe_table['Dist_Bad'].apply(lambda x: max(x, epsilon))
    
    # 4. Calculate WoE and IV
    woe_table['WoE'] = np.log(woe_table['Dist_Good'] / woe_table['Dist_Bad'])
    woe_table['IV'] = (woe_table['Dist_Good'] - woe_table['Dist_Bad']) * woe_table['WoE']
    
    information_value = woe_table['IV'].sum()
    
    # Add Feature Name to the table
    woe_table['Feature'] = feature
    
    logger.info(f"WoE/IV calculated for {feature}. IV: {information_value:.4f}")
    return woe_table, information_value


def transform_data_to_woe(data: pd.DataFrame, woe_lookup: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Transforms the continuous feature values in the input data to their calculated WoE values.

    Args:
        data (pd.DataFrame): The original DataFrame with continuous features.
        woe_lookup (pd.DataFrame): The combined lookup table containing bins and WoE values.
        features (List[str]): List of features to be transformed.

    Returns:
        pd.DataFrame: The DataFrame with WoE features replacing continuous features.
    """
    logger.info("Starting data transformation to WoE values...")
    df_woe = data.copy()
    
    for feature in features:
        feature_woe_map = {}
        lookup_subset = woe_lookup[woe_lookup['Feature'] == feature]
        
        # Iterate over the bins in the lookup table
        for index, row in lookup_subset.iterrows():
            # The 'Bin' column holds the Interval notation (e.g., (10.0, 20.0])
            interval = pd.Interval(row['Bin'].left, row['Bin'].right, closed=row['Bin'].closed)
            
            # Map the WoE value to the continuous feature data points if they fall in the bin
            df_woe.loc[df_woe[feature].apply(lambda x: x in interval), f'{feature}_WoE'] = row['WoE']

        # Drop the original continuous feature column
        df_woe = df_woe.drop(columns=[feature])
        
    logger.info("Data transformation complete. Features are now WoE scores.")
    return df_woe


def save_artifacts(df_woe: pd.DataFrame, woe_lookup: pd.DataFrame, data_path: str, lookup_path: str):
    """
    Saves the transformed data and the WoE lookup table artifacts.

    Args:
        df_woe (pd.DataFrame): The final WoE transformed DataFrame.
        woe_lookup (pd.DataFrame): The combined WoE lookup table.
        data_path (str): Path to save the transformed data CSV.
        lookup_path (str): Path to save the WoE lookup table CSV.
    """
    logger.info(f"Saving WoE transformed data to {data_path}")
    df_woe.to_csv(data_path, index=False)
    
    logger.info(f"Saving WoE lookup table to {lookup_path} (Artifact for deployment)")
    woe_lookup.to_csv(lookup_path, index=False)
    logger.info("WoE/IV transformation script finished successfully.")


# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        # A. Load data
        df = load_data(INPUT_PATH)
        
        all_woe_tables = []
        total_iv_summary = {}

        # B. Calculate WoE and IV for each feature
        for feature in FEATURES:
            woe_table, iv_value = calculate_woe_iv(df, feature, TARGET_COLUMN)
            if not woe_table.empty:
                all_woe_tables.append(woe_table)
                total_iv_summary[feature] = iv_value

        # C. Combine all WoE tables into a single lookup artifact
        if not all_woe_tables:
            logger.error("No valid WoE tables were generated. Aborting.")
            exit()
            
        woe_lookup_table = pd.concat(all_woe_tables, axis=0)
        
        logger.info("\n--- Information Value (IV) Summary ---")
        for feature, iv in total_iv_summary.items():
            logger.info(f"  {feature}: {iv:.4f} ({'Strong' if iv >= 0.3 else 'Moderate'})")
        logger.info("------------------------------------\n")

        # D. Transform data to WoE values
        df_transformed = transform_data_to_woe(df.drop(columns=['RFMS_Score']), woe_lookup_table, FEATURES)
        
        # E. Save artifacts
        save_artifacts(df_transformed, woe_lookup_table, OUTPUT_DATA_PATH, LOOKUP_TABLE_PATH)

    except Exception as e:
        logger.critical(f"Pipeline failed during WoE/IV transformation: {e}")