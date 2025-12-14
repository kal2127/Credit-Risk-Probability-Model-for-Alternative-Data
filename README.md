# Credit-Risk-Probability-Model-for-Alternative-Data

# Credit Risk Probability Model for Alternative Data

## 🎯 Project Goal

The objective of this project is to build a highly interpretable credit scorecard model for Bati Bank by leveraging alternative transactional data (Xente data) to assess customer risk. The model utilizes the Weight of Evidence (WoE) transformation to create a stable, regulatory-compliant Logistic Regression model.

## 📁 Project Structure

- `data/`: Contains the raw transactional data (`data.csv`) and the intermediate feature files.
- `notebooks/`: Contains the Python scripts and Jupyter notebooks for analysis and model training.
  - `feature_engineering.py`: Script to calculate RFMS features and define the Default Proxy.
  - `woe_iv_transformation.py`: Script to transform features using WoE/IV.
  - `model_building.py`: Script to train the Logistic Regression model and calibrate the scorecard.
  - `EDA_Feature_Engineering.ipynb`: **Missing EDA deliverable (to be completed).**
- `models/`: Stores the model artifacts required for deployment (e.g., WoE lookup tables, scorecard coefficients).

## 🚀 Setup and Run Instructions

### Prerequisites

1.  Python 3.8+
2.  Git

### Step 1: Clone the Repository

```bash
git clone [Insert your project repository URL here]
cd credit_risk_project
Credit Scoring Business Understanding
1. Basel II Accord and Model Interpretability

The Basel II Accord requires banks to carefully measure and manage credit risk, especially when using internal models to make lending decisions. Because these models directly affect regulatory capital and customer outcomes, they must be transparent, interpretable, and well documented.

An interpretable model allows regulators, auditors, and internal risk teams to clearly understand how risk is calculated and to verify that the model is working as intended. It also enables the bank to explain credit decisions to customers in a clear and fair way. For these reasons, Basel II strongly encourages the use of models that can be easily validated, audited, and justified rather than complex models that cannot be clearly explained.

2. Proxy Variable and Its Business Risks

This project uses alternative data from an eCommerce platform, which does not include a direct indicator of loan default (such as missed or late repayments). Since customers are applying for a new Buy-Now-Pay-Later service, there is no historical loan performance data available.

To address this limitation, a proxy target variable is created using customer transaction behavior, such as Recency, Frequency, and Monetary (RFM) patterns. These behavioral indicators serve as a substitute for a true default label and allow the model to estimate customer risk.

However, using a proxy variable introduces important business risks. The proxy may not perfectly represent actual credit risk, which can lead to rejecting creditworthy customers or approving risky ones. Additionally, behavioral data may unintentionally capture bias, raising fairness and regulatory concerns. Finally, a proxy designed for one eCommerce platform may not generalize well to other financial products or lending contexts.

3. Trade-offs Between Simple and Complex Models

In a regulated financial environment, there is a trade-off between model interpretability and predictive performance.

Simple models such as Logistic Regression with Weight of Evidence (WoE) are widely used in credit scoring because they are easy to understand and explain. Their outputs can be directly interpreted, making them suitable for regulatory review and customer communication. However, these models may have lower predictive accuracy when dealing with complex, non-linear patterns in transactional data.

More complex models such as Gradient Boosting often achieve higher predictive performance by capturing complex relationships in the data. Despite their accuracy, they are harder to interpret and are often considered “black-box” models. This lack of transparency makes regulatory approval more challenging unless additional explainability tools are applied.

As a result, simpler and more interpretable models are typically preferred for deployment in regulated banking environments, while complex models are often used for benchmarking or as supporting models.
```
