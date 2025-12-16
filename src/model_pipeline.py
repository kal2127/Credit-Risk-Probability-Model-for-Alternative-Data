import pandas as pd
import numpy as np
import joblib
import mlflow
import warnings

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Suppress sklearn/MLflow warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. SIMULATE RAW DATA ---
def load_and_simulate_raw_data(num_samples=5000):
    """
    Simulates raw transaction data structure including CustomerID, Amount, and Time.
    """
    np.random.seed(42)
    start_date = pd.to_datetime('2023-01-01')
    end_date = pd.to_datetime('2023-12-31')
    
    # Generate customer IDs and transaction data
    customer_ids = [f'CUST_{i}' for i in range(num_samples // 5)]
    data = []
    
    # Ensure some customers have high activity and some are low (for clear clustering)
    high_activity_ids = np.random.choice(customer_ids, size=len(customer_ids)//5, replace=False)
    
    for _ in range(num_samples):
        customer_id = np.random.choice(customer_ids)
        
        # Make high activity customers spend more/more often
        if customer_id in high_activity_ids:
            amount = np.random.lognormal(mean=6, sigma=0.5)
            time_offset = np.random.randint(0, (end_date - start_date).days * 24 * 60 * 60)
        else:
            amount = np.random.lognormal(mean=4, sigma=1.0)
            time_offset = np.random.randint(0, (end_date - start_date).days * 24 * 60 * 60)
        
        transaction_time = start_date + pd.Timedelta(seconds=time_offset)
        
        data.append([customer_id, amount, transaction_time])
        
    df = pd.DataFrame(data, columns=['CustomerID', 'TransactionAmount', 'TransactionStartTime'])
    df.sort_values('TransactionStartTime', inplace=True)
    return df

# --- 2. CORE TRANSFORMATION FUNCTIONS (Pipeline Steps) ---

def calculate_rfms(df):
    """Calculates Recency, Frequency, Monetary Value, and Std Deviation (RFMS)."""
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    
    rfm_df = df.groupby('CustomerID').agg(
        Recency_Days=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
        Frequency=('CustomerID', 'count'),
        Monetary_Value=('TransactionAmount', 'mean'),
        Std_Deviation=('TransactionAmount', 'std')
    ).fillna(0).reset_index() # Fill NaN std (single transactions) with 0
    
    return rfm_df

def extract_temporal_features(df):
    """
    Extracts explicit temporal features (Transaction Hour/Month/DayOfWeek)
    based on the *last* transaction for each customer.
    """
    # Find the last transaction time for each customer
    last_transaction_time = df.groupby('CustomerID')['TransactionStartTime'].max().reset_index()
    
    # Merge RFMS features into this temporal dataframe for a complete view
    rfms_data = calculate_rfms(df)
    temp_df = last_transaction_time.merge(rfms_data[['CustomerID']], on='CustomerID', how='left')

    temp_df['TransactionHour'] = last_transaction_time['TransactionStartTime'].dt.hour
    temp_df['TransactionMonth'] = last_transaction_time['TransactionStartTime'].dt.month
    temp_df['TransactionDayOfWeek'] = last_transaction_time['TransactionStartTime'].dt.dayofweek
    
    # Only return the required features for the pipeline
    return temp_df[['CustomerID', 'TransactionHour', 'TransactionMonth', 'TransactionDayOfWeek']]

# --- 3. PROXY TARGET DEFINITION (RFM K-Means Clustering) ---

def create_target_variable(df_raw):
    """
    Implements K-Means clustering on RFM features to define the high-risk proxy (Task 4 Improvement).
    """
    # 1. Calculate RFM features
    df_rfms = calculate_rfms(df_raw)
    
    # 2. Select and scale RFM features for clustering
    rfm_features = df_rfms[['Recency_Days', 'Frequency', 'Monetary_Value']]
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_features)

    # 3. K-Means Clustering into 3 groups
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_rfms['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # 4. Analyze clusters to identify the high-risk segment
    cluster_profile = df_rfms.groupby('Cluster')[rfm_features.columns].mean()
    print("--- RFM Cluster Profiles (Mean Values) ---")
    print(cluster_profile)
    
    # Identify the high-risk cluster (High Recency, Low Frequency, Low Monetary)
    # This cluster represents the least engaged segment.
    low_engagement_cluster = cluster_profile.sort_values(
        by=['Recency_Days', 'Frequency', 'Monetary_Value'], 
        ascending=[False, True, True]
    ).index[0]
    
    print(f"\nIdentified High-Risk Cluster: {low_engagement_cluster}")

    # 5. Create the binary target column
    df_rfms['is_high_risk'] = np.where(df_rfms['Cluster'] == low_engagement_cluster, 1, 0)
    
    return df_rfms[['CustomerID', 'is_high_risk']]


# --- 4. MODEL EVALUATION (Full set of metrics) ---

def evaluate_model(y_true, y_pred_proba, y_pred):
    """Calculates all required metrics (Task 5)."""
    # KS Statistic calculation (for credit risk reporting)
    y_true_sorted, y_pred_proba_sorted = zip(*sorted(zip(y_true, y_pred_proba), key=lambda x: x[1]))
    y_true_sorted = np.array(y_true_sorted)
    
    # Cumulative proportion of goods (0s) and bads (1s)
    cpg = np.cumsum(y_true_sorted == 0) / np.sum(y_true_sorted == 0)
    cpb = np.cumsum(y_true_sorted == 1) / np.sum(y_true_sorted == 1)
    ks = np.max(np.abs(cpg - cpb))
    
    metrics = {
        'AUC': roc_auc_score(y_true, y_pred_proba),
        'GINI': 2 * roc_auc_score(y_true, y_pred_proba) - 1,
        'KS_Statistic': ks,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1_Score': f1_score(y_true, y_pred)
    }
    return metrics

# --- 5. EXECUTION AND MLFLOW TRACKING ---

def run_mlops_pipeline():
    """Main function to run the complete MLOps pipeline, including tuning and MLflow."""
    
    # 1. Load and simulate raw data
    raw_df = load_and_simulate_raw_data()
    
    # 2. Feature Engineering and Target Creation
    
    # Create the K-Means based target variable (is_high_risk)
    target_df = create_target_variable(raw_df.copy())
    
    # Calculate all RFMS features
    rfms_data = calculate_rfms(raw_df)
    
    # Extract all Temporal features
    temporal_data = extract_temporal_features(raw_df)
    
    # Merge all features and the target into the final dataset
    model_df = rfms_data.merge(temporal_data, on='CustomerID', how='inner')
    model_df = model_df.merge(target_df, on='CustomerID', how='inner')
    
    # Prepare final X and y
    X = model_df.drop(columns=['CustomerID', 'is_high_risk'])
    y = model_df['is_high_risk']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\nData ready. Training samples: {X_train.shape[0]}, Target balance (1s): {y.mean():.2f}")
    
    # --- 3. SKLEARN PIPELINE DEFINITION ---
    
    # Define feature groups based on how they are treated
    rfms_features = ['Recency_Days', 'Frequency', 'Monetary_Value', 'Std_Deviation']
    temporal_numeric = ['TransactionHour'] 
    temporal_categorical = ['TransactionMonth', 'TransactionDayOfWeek'] 

    # Preprocessor definition (handles scaling and encoding)
    preprocessor = ColumnTransformer(
        transformers=[
            # 1. Scaling for all main RFMS features and continuous temporal features
            ('scaler', StandardScaler(), rfms_features + temporal_numeric),
            
            # 2. One-Hot Encoding for categorical temporal features 
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), temporal_categorical)
        ],
        remainder='passthrough'
    )
    
    # --- 4. MODEL TRAINING, TUNING, AND MLFLOW ---
    
    # Define models and their hyperparameter grids
    models_to_train = {
        'LogisticRegression': (LogisticRegression(solver='liblinear', random_state=42), 
                               {'classifier__C': [0.01, 0.1, 1]}),
        'XGBClassifier': (XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                          {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [3, 5]}),
        'LGBMClassifier': (LGBMClassifier(random_state=42),
                           {'classifier__n_estimators': [50, 100], 'classifier__num_leaves': [10, 20]})
    }

    # MLflow Setup
    mlflow.set_experiment("Credit_Risk_Scorecard_Optimization")
    
    best_auc = 0
    best_model_name = ""
    best_pipeline = None

    for model_name, (model, param_grid) in models_to_train.items():
        # Start a new MLflow run for each model type
        with mlflow.start_run(run_name=f"Tuning_{model_name}") as run:
            
            # Create a full pipeline for the current model
            full_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            # Hyperparameter Tuning with GridSearchCV
            print(f"\nStarting Grid Search for {model_name}...")
            grid_search = GridSearchCV(
                full_pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            best_estimator = grid_search.best_estimator_
            
            # Predict on Test Set
            y_pred_proba = best_estimator.predict_proba(X_test)[:, 1]
            y_pred = best_estimator.predict(X_test)
            
            # Evaluate (Full Metric Set)
            metrics = evaluate_model(y_test, y_pred_proba, y_pred)
            
            # MLflow Logging
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("target_definition", "RFM K-Means Cluster (k=3)")
            
            # Log the model artifact
            mlflow.sklearn.log_model(best_estimator, f"model_{model_name}")
            
            print(f"--- {model_name} Results ---")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
            
            if metrics['AUC'] > best_auc:
                best_auc = metrics['AUC']
                best_model_name = model_name
                best_pipeline = best_estimator

    # 5. Final Model Selection and Saving
    print(f"\nMLflow tracking complete. Best model is: {best_model_name} (AUC: {best_auc:.4f})")
    
    # Save the best pipeline for deployment
    model_filepath = 'models/best_model_pipeline.pkl'
    # Ensure the 'models' directory exists before saving
    import os
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_pipeline, model_filepath)
    print(f"Best model saved to {model_filepath}")
    
    # Register the best model (Simulated)
    with mlflow.start_run(run_name="Final_Best_Model_Registration") as run:
        mlflow.sklearn.log_model(best_pipeline, "production_pipeline")
    
if __name__ == "__main__":
    run_mlops_pipeline()