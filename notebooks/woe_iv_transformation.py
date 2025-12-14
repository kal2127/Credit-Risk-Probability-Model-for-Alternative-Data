import pandas as pd
import numpy as np
from pathlib import Path

# --- Configuration (using robust path resolution) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- File Paths ---
INPUT_FILE = PROJECT_ROOT / 'data' / 'rfms_features_with_proxy.csv'
OUTPUT_FILE = PROJECT_ROOT / 'data' / 'rfms_woe_transformed_data.csv'
# Using the models directory to save the lookup table, which is critical for deployment
WOE_LOOKUP_PATH = PROJECT_ROOT / 'models' / 'woe_lookup_table.csv'

# --- Variables ---
TARGET_COLUMN = 'Default_Proxy'
FEATURES = ['Recency', 'Frequency', 'Monetary_Value', 'Std_Deviation']
# Number of bins to use for the transformation
NUM_BINS = 5


def load_data(file_path):
    """Loads the RFMS data with the proxy target."""
    try:
        df = pd.read_csv(file_path)
        print(f"RFMS data loaded successfully with {len(df)} records.")
        return df
    except FileNotFoundError:
        print(f"ERROR: RFMS feature file not found at {file_path}. Please run feature_engineering.py first.")
        return None

def calculate_woe_iv(df, feature, target):
    """
    Calculates WoE and IV for a binned feature.
    """
    # Group by the bin and calculate counts
    grouped = df.groupby(feature)[target].agg(
        Total_Count=('count'),
        Bad_Count=('sum')
    ).reset_index()
    
    # Good count is Total - Bad
    grouped['Good_Count'] = grouped['Total_Count'] - grouped['Bad_Count']

    # Total Good and Bad Counts across the whole dataset
    total_good = grouped['Good_Count'].sum()
    total_bad = grouped['Bad_Count'].sum()
    
    # Calculate distributions
    grouped['%_Good'] = grouped['Good_Count'] / total_good
    grouped['%_Bad'] = grouped['Bad_Count'] / total_bad
    
    # Calculate WoE (Weight of Evidence)
    # Added a small epsilon (1e-6) for stability in case of division by zero (no Good/Bad in a bin)
    grouped['WoE'] = np.log((grouped['%_Good'] + 1e-6) / (grouped['%_Bad'] + 1e-6))
    
    # Calculate IV (Information Value)
    grouped['IV_Contrib'] = (grouped['%_Good'] - grouped['%_Bad']) * grouped['WoE']
    
    # Calculate total IV for the feature
    total_iv = grouped['IV_Contrib'].sum()
    
    # Add feature name and total IV to the summary
    grouped['Feature'] = feature
    
    return grouped[['Feature', feature, 'Total_Count', 'Bad_Count', 'WoE', 'IV_Contrib']], total_iv

def perform_woe_iv_transformation(df, features, target):
    """
    Bins the data, calculates WoE/IV for all features, and transforms the data.
    """
    woe_lookup_list = []
    iv_list = []
    transformed_df = df[[target]].copy()

    for feature in features:
        print(f"\nProcessing feature: {feature}")
        
        # 1. Optimal Binning (using Quantiles as a robust proxy for optimal binning)
        # Bins the continuous feature into discrete groups
        try:
            df[f'{feature}_Binned'] = pd.qcut(df[feature], q=NUM_BINS, labels=False, duplicates='drop')
        except ValueError:
            print(f"WARNING: Could not create {NUM_BINS} distinct bins for {feature}. Using smaller number of bins.")
            # Fallback to a lower number of bins if data is too concentrated
            df[f'{feature}_Binned'] = pd.qcut(df[feature], q=3, labels=False, duplicates='drop')


        # 2. Calculate WoE and IV for the binned feature
        woe_table, total_iv = calculate_woe_iv(df, f'{feature}_Binned', target)
        woe_lookup_list.append(woe_table)
        iv_list.append({'Feature': feature, 'IV': total_iv})
        
        # 3. WoE Transformation (Map WoE values back to the original DataFrame)
        woe_mapping = woe_table.set_index(f'{feature}_Binned')['WoE'].to_dict()
        transformed_df[f'{feature}_WoE'] = df[f'{feature}_Binned'].map(woe_mapping)

    # Compile IV Summary
    iv_summary = pd.DataFrame(iv_list).sort_values(by='IV', ascending=False)
    
    print("\n--- Information Value (IV) Summary ---")
    print(iv_summary.to_string(index=False))
    
    # Select features with strong predictive power (IV > 0.1)
    # The IV threshold is a guideline; for a small set of RFMS features, we often use all.
    strong_features = iv_summary[iv_summary['IV'] > 0.05]['Feature'].tolist()
    
    if not strong_features:
        print("\n[WARNING] No features passed the IV > 0.05 threshold. Using all features.")
        strong_features = features
        
    print(f"\nFeatures selected for modeling (IV > 0.05): {strong_features}")
    
    # Compile the final WoE lookup table
    woe_lookup_final = pd.concat(woe_lookup_list)
    
    # Filter the transformed data to keep only the selected features
    selected_woe_columns = [f'{col}_WoE' for col in strong_features]
    final_transformed_df = transformed_df[[target] + selected_woe_columns]
    
    return final_transformed_df, woe_lookup_final


def main():
    """Main function to run the WoE/IV transformation pipeline."""
    
    # 1. Load data
    df = load_data(INPUT_FILE)
    if df is None:
        return

    # 2. Perform WoE/IV transformation
    woe_df, woe_lookup = perform_woe_iv_transformation(df, FEATURES, TARGET_COLUMN)

    # 3. Save the final dataset for modeling
    woe_df.to_csv(OUTPUT_FILE, index=False)
    woe_lookup.to_csv(WOE_LOOKUP_PATH, index=False)
    
    print(f"\nWoE transformed features saved to {OUTPUT_FILE}")
    print(f"WoE lookup tables (CRITICAL for deployment) saved to {WOE_LOOKUP_PATH}")


if __name__ == "__main__":
    main()