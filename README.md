# 💳 Credit Card Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B)](https://streamlit.io/)  
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-yellow)](https://xgboost.readthedocs.io/)  
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-lightblue)](https://pandas.pydata.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  

An interactive **Streamlit app** for **credit card fraud detection**,  
powered by **XGBoost**, capable of predicting whether transactions are **fraudulent** or **legitimate**.  
Supports **single or multiple CSV transactions** and returns both **predictions** and **fraud probabilities**.

---

## 🚀 Features

- **End-to-end ML Pipeline**
  - Data preprocessing (handling missing values, scaling)
  - Class imbalance handling using **SMOTE**
  - XGBoost classifier training
  - Model evaluation
- **Streamlit Frontend**
  - Upload CSV transactions
  - Display predictions and fraud probabilities
- **Flexible Input**
  - Single transaction as dict
  - Multiple transactions as list of dicts or CSV
- **Outputs**
  - `Prediction` (0 = Legitimate, 1 = Fraud)
  - `Fraud_Prob` (probability of fraud)

---

## 📂 CSV Format

Your input CSV **must** include at least these columns:

- `Time` — Timestamp of transaction  
- `V1` to `V28` — Anonymized features  
- `Amount` — Transaction amount  
- `Class` — Optional (used for evaluation, 0 = Legit, 1 = Fraud)  

Example:

| Time  | V1     | V2    | ... | V28   | Amount  | Class |
|-------|--------|-------|-----|-------|---------|-------|
| 0     | -1.36  | -0.07 | ... | -0.02 | 149.62  | 0     |
| 1     | 1.19   | 0.26  | ... | 0.01  | 2.69    | 0     |

---

## 🛠 Installation

1. Clone the repository:  
   `git clone git@github.com:Kumar-gaurav-rvce/credit-card-fraud-detection.git`  
   `cd credit-card-fraud-detection`  

2. Install Python dependencies:  
   `pip install -r requirements.txt`

---

## ▶ Run Locally

Run the Streamlit app:

```bash
streamlit run app.py
```

## 📜 License

This project is licensed under the [MIT License](./LICENSE).  
You are free to use, modify, and distribute this software under the terms of the MIT License.
