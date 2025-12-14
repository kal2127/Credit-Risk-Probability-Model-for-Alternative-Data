import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# --- 1. SETUP LOGGING ---
LOG_FILE = 'model_building.log'
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(LOG_FILE, mode='w'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# --- 2. CONFIGURATION ---
INPUT_DATA_PATH = '../data/rfms_woe_transformed_data.csv'
TARGET_COLUMN = 'Default_Proxy'
OUTPUT_SCORECARD_PATH = '../models/scorecard_coefficients.csv'
BASE_SCORE = 500  # Starting score for the scorecard
PDO = 20         # Points to Double the Odds
ODDS = 50 / 1    # Odds at the Base Score (e.g., 50:1)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the WoE transformed data for model training.

    Args:
        file_path (str): The path to the WoE transformed CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    logger.info(f"Loading WoE transformed data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Total observations: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def prepare_data(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Separates features and target, and splits the data into training and testing sets.

    Args:
        df (pd.DataFrame): The DataFrame with WoE features and the target.
        target (str): The name of the target column.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: 
            X_train, X_test, y_train, y_test.
    """
    # Select only the WoE features
    features = [col for col in df.columns if col.endswith('_WoE')]
    
    X = df[features]
    y = df[target]
    
    # Stratified split to maintain proxy distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    logger.info(f"Data split into Train ({len(X_train)}) and Test ({len(X_test)}) sets.")
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """
    Trains the Logistic Regression model.

    Args:
        X_train (pd.DataFrame): Training features (WoE values).
        y_train (pd.Series): Training target (Default_Proxy).

    Returns:
        LogisticRegression: The trained model object.
    """
    logger.info("Starting Logistic Regression model training...")
    # C=100 is chosen to prevent overly strong regularization (L2 penalty is default)
    model = LogisticRegression(C=100, penalty='l2', solver='liblinear', random_state=42)
    model.fit(X_train, y_train)
    logger.info("Model training complete.")
    return model


def evaluate_model(model: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluates the model and calculates key credit risk metrics.

    Args:
        model (LogisticRegression): The trained model.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target.

    Returns:
        Dict[str, float]: A dictionary of performance metrics (AUC, GINI, KS).
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 1. AUC (Area Under the Curve)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # 2. GINI Coefficient (2 * AUC - 1)
    gini = 2 * auc - 1
    
    # 3. Kolmogorov-Smirnov (KS) Statistic
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    ks_stat = np.max(tpr - fpr)
    
    metrics = {
        'AUC': auc,
        'GINI': gini,
        'KS_Statistic': ks_stat
    }
    
    logger.info("\n--- Model Performance Metrics (Test Set) ---")
    logger.info(f"AUC: {metrics['AUC']:.4f}")
    logger.info(f"GINI: {metrics['GINI']:.4f} ({'Strong' if metrics['GINI'] >= 0.4 else 'Acceptable'})")
    logger.info(f"KS Statistic: {metrics['KS_Statistic']:.4f}")
    logger.info("------------------------------------------")
    
    return metrics


def calculate_scorecard(model: LogisticRegression, X_train: pd.DataFrame) -> pd.DataFrame:
    """
    Calibrates the Logistic Regression model into a score card based on business parameters.
    
    Score = (A) + (B * WoE)
    where A = Base Score - (Factor * Log(Odds))
    and B = Factor = PDO / Log(2)

    Args:
        model (LogisticRegression): The trained Logistic Regression model.
        X_train (pd.DataFrame): The training features used for coefficient extraction.

    Returns:
        pd.DataFrame: A DataFrame containing the final scorecard coefficients (A, B, Base).
    """
    logger.info(f"Calibrating model into Scorecard (Base Score: {BASE_SCORE}, PDO: {PDO}, Odds: {ODDS}:1)...")
    
    # Log Odds at Base Score
    log_odds = np.log(ODDS) 
    
    # Factor (B): Determines how many points the score changes per unit of WoE
    factor = PDO / np.log(2)
    
    # Offset (A): Determines the base score intercept
    offset = BASE_SCORE - (factor * log_odds)
    
    # The intercept of the model (a model parameter)
    model_intercept = model.intercept_[0]
    
    # The coefficients of the model (WoE values)
    model_coefficients = model.coef_[0]

    # Create the scorecard lookup table
    scorecard_df = pd.DataFrame({
        'Feature_WoE': X_train.columns,
        'Coefficient': model_coefficients
    })
    
    # Add Base Score and Point per WoE calculations
    scorecard_df['Points_per_WoE'] = scorecard_df['Coefficient'] * (-factor)
    
    # Add the scorecard parameters to the output
    scorecard_params = {
        'Feature_WoE': ['Model_Intercept', 'Scorecard_Offset', 'Scorecard_Factor'],
        'Coefficient': [model_intercept, offset, factor],
        'Points_per_WoE': [np.nan, np.nan, np.nan] # Not applicable for these rows
    }
    scorecard_df = pd.concat([scorecard_df, pd.DataFrame(scorecard_params)])

    logger.info(f"Scorecard calibration complete. Factor (B): {factor:.2f}. Offset (A): {offset:.2f}.")
    return scorecard_df


def plot_score_distribution(model: LogisticRegression, X: pd.DataFrame, y: pd.Series):
    """
    Generates and saves the score distribution plot between Good (0) and Bad (1) customers.
    This is a critical visualization for the final report. 

    Args:
        model (LogisticRegression): The trained model.
        X (pd.DataFrame): The features.
        y (pd.Series): The target.
    """
    logger.info("Generating Score Distribution Plot...")
    
    # Predict the log-odds (logit)
    y_pred_logit = model.decision_function(X)
    
    # Calculate the final score using the calibration formula
    log_odds = np.log(ODDS)
    factor = PDO / np.log(2)
    offset = BASE_SCORE - (factor * log_odds)
    
    # Score = Offset + (Factor * Logit)
    final_score = offset + (-factor * y_pred_logit)

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Filter scores for Good (0) and Bad (1) customers
    good_scores = final_score[y == 0]
    bad_scores = final_score[y == 1]
    
    sns.histplot(good_scores, bins=30, kde=True, color='green', label='Good Customers (0)', stat='density', line_kws={'linewidth': 2})
    sns.histplot(bad_scores, bins=30, kde=True, color='red', label='Bad Customers (1)', stat='density', line_kws={'linewidth': 2})
    
    plt.title('Credit Score Distribution: Good vs. Bad Customers')
    plt.xlabel('Scorecard Score')
    plt.ylabel('Density')
    plt.legend()
    
    plot_path = '../reports/score_distribution.png'
    # Ensure the reports directory exists for saving the plot
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    logger.info(f"Score distribution plot saved to {plot_path}")
    plt.close()


def save_scorecard(df: pd.DataFrame, file_path: str):
    """
    Saves the final scorecard coefficients to a CSV file.

    Args:
        df (pd.DataFrame): The scorecard DataFrame.
        file_path (str): The destination path.
    """
    logger.info(f"Saving final scorecard coefficients to {file_path}")
    df.to_csv(file_path, index=False)
    logger.info("Model building script finished successfully.")


# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    import seaborn as sns # Import here for plotting function access
    
    try:
        # A. Load data
        df = load_data(INPUT_DATA_PATH)
        
        # B. Prepare and split data
        X_train, X_test, y_train, y_test = prepare_data(df, TARGET_COLUMN)
        
        # C. Train model
        model = train_model(X_train, y_train)
        
        # D. Evaluate model and log metrics
        evaluate_model(model, X_test, y_test)
        
        # E. Calculate Scorecard
        scorecard_df = calculate_scorecard(model, X_train)
        
        # F. Plot Score Distribution (using all data for a comprehensive view)
        plot_score_distribution(model, df[[col for col in df.columns if col.endswith('_WoE')]], df[TARGET_COLUMN])
        
        # G. Save artifacts
        save_scorecard(scorecard_df, OUTPUT_SCORECARD_PATH)

    except Exception as e:
        logger.critical(f"Pipeline failed during model building: {e}")