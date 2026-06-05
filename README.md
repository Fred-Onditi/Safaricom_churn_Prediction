# Telecom & Mobile Money Customer Churn Prediction Engine

This repository contains an end-to-end machine learning pipeline built to predict customer churn using telecom performance metrics and mobile money transaction data. 

## Key Highlights & Machine Learning Architecture
- **Production-Ready Pipeline:** Combines preprocessing (`SimpleImputer`, `OneHotEncoder`), class rebalancing, and classification into a single, leak-proof Scikit-Learn `Pipeline`.
- **Advanced Class Imbalance Handling:** Implements **SMOTE** (Synthetic Minority Over-sampling Technique) via `imblearn` to handle a severe 12.38% minority class imbalance without overfitting.
- **Precision-Recall Threshold Optimization:** Bypassed the standard 0.50 classification default to locate a mathematically optimal threshold of **0.2310**, boosting churn target **Recall from 5% to 61%**.

## Performance Metrics (Optimized Operational Threshold)
- **ROC AUC Score:** 0.7780
- **Optimized F1-Score:** 0.42
- **Class 1 (Churned) Recall:** 61% (Catches nearly 2/3 of all at-risk customers)
- **Class 1 (Churned) Precision:** 32% (Nearly triples predictive accuracy over random guessing)

### Confusion Matrix
| | Predicted Loyal | Predicted Churn Risk |
|---|---|---|
| **Actual Loyal** | 712 (True Negatives) | 164 (False Positives) |
| **Actual Churned**| 48 (False Negatives) | 76 (True Positives) |

## Top Data-Driven Business Insights
Based on the Random Forest feature importance extraction, the top operational drivers of customer churn are:
1. **Days Since Last Topup:** The strongest behavioral indicator of a customer shifting to a competitor.
2. **Customer Service Calls:** Highlights structural or network frustration compounding before a customer leaves.
3. **Has Safaricom Home:** Multi-service ecosystem lock-in drastically increases customer retention.

##  How to Use the Saved Model
```python
import joblib

# Load the production-ready pipeline
model = joblib.load('safaricom_churn_model.pkl')

# Generate risk probabilities on new customer data
probabilities = model.predict_proba(new_customer_data)[:, 1]
is_at_risk = (probabilities >= 0.2310).astype(int)
