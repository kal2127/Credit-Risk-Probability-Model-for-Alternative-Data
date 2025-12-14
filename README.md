Credit Risk Probability Model for Alternative Data

I. Project Overview and Context
This project develops a stable, regulatory-compliant credit scorecard model for Bati Bank. It leverages alternative transactional data from the Xente platform to assess customer risk for a new Buy-Now-Pay-Later (BNPL) service.

🎯 Project Goal
The primary objective is to build a highly interpretable credit scorecard using Logistic Regression based on Weight of Evidence (WoE) transformed features, ensuring the model meets high standards for stability, performance, and auditability.

A. Business Understanding and Regulatory Context

1. Basel II Accord and Model Interpretability
   The use of this model for lending decisions requires strict adherence to the Basel II Accord. This mandates that the model must be transparent and auditable.

Requirement: We use Logistic Regression and WoE transformation because they provide easily explainable outputs (scorecard points) that satisfy regulatory demands for transparency.

Purpose: Allows regulators and auditors to clearly understand how risk is calculated and enables clear explanation of credit decisions to customers.

2. Proxy Variable and Associated Business Risks
   Since historical loan performance data is unavailable for this new service, a Default Proxy is created based on customer transaction behavior (RFMS patterns).

Limitation: The proxy is a substitute for a true default label, carrying inherent risks:

Inaccuracy: The proxy may not perfectly reflect actual credit risk.

Bias: Behavioral data may unintentionally capture bias, raising fairness concerns.

3. Trade-offs: Simple vs. Complex Models
   In a regulated financial environment, the WoE Logistic Regression model (Simple) is chosen over "black-box" alternatives (Complex) because interpretability is prioritized over marginal predictive gains, satisfying regulatory requirements.

II. Technical Setup
📁 Project Structure
The repository follows a standard layout, separating source code from analysis and configuration.

data/: Contains the raw transactional data (data.csv). (Excluded from Git by .gitignore)

src/: Contains the executable Python scripts for the entire pipeline.

feature_engineering.py: Calculates RFMS features.

woe_iv_transformation.py: Transforms features using WoE/IV.

model_building.py: Trains the final Logistic Regression model.

notebooks/: Contains the Jupyter Notebooks for analysis and documentation.

EDA_Feature_Engineering.ipynb: The required EDA deliverable.

models/: Stores the final model artifacts (WoE lookup tables, scorecard coefficients).

tests/: Contains unit tests for key pipeline functions.

Prerequisites
Python 3.8+

Git

Step 1: Clone the Repository
Bash

git clone [Insert your project repository URL here]
cd credit_risk_project
Step 2: Set up Environment and Dependencies
A virtual environment is required. All necessary packages are listed in requirements.txt.

Bash

# Set up virtual environment (Linux/macOS example)

python3 -m venv venv
source venv/bin/activate

# Install dependencies

pip install -r requirements.txt
III. Pipeline Execution
The pipeline must be executed sequentially from the project root directory (credit_risk_project/). All scripts utilize Python's logging module for production monitoring and debugging.

Step 1: Feature Engineering (Task 2.A)
Aggregates transactional data into RFMS features and defines the Default Proxy.

Bash

python src/feature_engineering.py
Step 2: WoE/IV Transformation (Task 2.B)
Bins the features, calculates WoE/IV, and transforms the data for Logistic Regression.

Bash

python src/woe_iv_transformation.py
Step 3: Model Building and Evaluation (Task 3)
Trains the model, calculates AUC/GINI/KS, calibrates the final credit scorecard, and saves the scoring coefficients.

Bash

python src/model_building.py
