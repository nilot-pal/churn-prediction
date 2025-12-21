# Customer Churn Prediction

## Problem Statement
Customer churn is a critical business problem where the goal is to identify customers at risk of leaving a service. Retaining existing customers is often significantly cheaper than acquiring new ones, making churn prediction a valuable use case for machine learning.

This project builds an end-to-end churn prediction pipeline, focusing not only on model performance but also on interpretability, evaluation, and deployment considerations.

---

## Dataset
The dataset contains customer-level information including:
- Contract type
- Tenure
- Monthly charges
- Payment method
- Service subscriptions

The target variable is **Churn**, indicating whether a customer left the service.

---

## Approach
The project follows an industry-style workflow:
1. Exploratory data analysis to understand churn drivers
2. Data cleaning and preprocessing using sklearn pipelines
3. Baseline modeling with Logistic Regression and Decision Trees
4. Ensemble modeling using Random Forest
5. Model evaluation using ROC and Precision–Recall curves
6. Threshold tuning based on business cost considerations
7. Error analysis and deployment-oriented conclusions

---

## Key Findings
- Short tenure and month-to-month contracts are the strongest drivers of churn.
- Logistic Regression provides a strong and interpretable baseline.
- Random Forest does not significantly outperform Logistic Regression, suggesting the churn signal is largely monotonic.
- Precision–Recall analysis shows that threshold selection has a greater impact on operational performance than model complexity.
- Default classification thresholds (0.5) are often suboptimal in cost-sensitive churn settings.

---

## Deployment Considerations
- Classification thresholds should be tuned using business costs such as customer lifetime value and intervention cost.
- Simpler models may be preferred when performance is comparable due to interpretability and stability.
- More complex models can be maintained as challenger models to detect evolving non-linear behavior.

---

## Technologies Used
- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

---

## Repository Structure
churn-prediction/
├── data/
├── notebooks/
│ └── 01_eda.ipynb
├── src/
├── README.md
