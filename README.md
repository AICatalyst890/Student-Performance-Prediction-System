# 🎓 Student Performance Prediction System

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete end-to-end Machine Learning project built to analyze student performance factors and predict final exam scores (`Exam_Score`) using linear, tree-based, and ensemble regression models.

---

## 📌 Project Overview

This project analyzes key academic, socio-economic, and behavioral factors influencing student test scores. Using `dataset.csv`, the pipeline cleans missing values, filters unrepresentative outliers, scales continuous features, encodes categorical variables, and evaluates multiple regression algorithms to identify the optimal predictive model.

---

## 📂 Dataset & Feature Processing

### 1. Missing Value Imputation
* **Numerical Features:** Imputed using `SimpleImputer(strategy='mean')`.
* **Categorical Features:** Imputed using `SimpleImputer(strategy='most_frequent')`.

### 2. Outlier Removal
* Filtered extreme values in tutoring frequency by retaining records where `Tutoring_Sessions < 8`.

### 3. Feature Transformations & Encoding
* **Nominal Features:** Transformed via `OneHotEncoder(drop='first', sparse_output=False)` to prevent multicollinearity.
* **Feature Scaling:** Applied `StandardScaler` (`X_train_scaled`, `X_test_scaled`) for distance-sensitive algorithms like SVR and KNN.
* **Target Management:** Re-indexed `Exam_Score` cleanly as the final target variable column.

---

## ⚙️ Model Architecture & Evaluation

The notebook evaluates a broad suite of linear, regularized, tree-based, and ensemble models:

* **Linear & Regularized Models:** `Linear Regression`, `Lasso`, `Ridge`, `ElasticNet`
* **Support Vector Machine:** `SVR` (trained on standardized features)
* **Tree-Based Models:** `Random Forest`, `XGBoost`
* **Ensemble Blending:** `VotingRegressor` combining Linear Regression, Random Forest, XGBoost, and SVR

### Evaluation Loop Setup

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

models_dict = {
    "Linear Regression": lr_model, 
    "Lasso": lasso_model, 
    "Ridge": ridge_model,
    "ElasticNet": elastic_model,
    "Random Forest": rf,
    "XGBoost": xgb_model,
    "Voting Regressor": voting_model,
    "SVR": svr_model
}

# Dynamic feature routing for scaled vs unscaled models
scaled_models = {"SVR", "KNN"}

for name, model in models_dict.items():
    X_tr = X_train_scaled if name in scaled_models else X_train
    X_te = X_test_scaled if name in scaled_models else X_test

    y_pred_train = model.predict(X_tr)
    y_pred_test = model.predict(X_te)
    
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = root_mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)

    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = root_mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    print("-" * 45)
    print(f"{name} Model Performance:")
    print("-" * 45)
    print(f"Train - MAE: {mae_train:.2f} | RMSE: {rmse_train:.2f} | R²: {r2_train:.4f}")
    print(f"Test  - MAE: {mae_test:.2f}  | RMSE: {rmse_test:.2f}  | R²: {r2_test:.4f}\n")

```

---

## 🛠️ Project Structure

```text
├── dataset.csv          # Source dataset
├── ml.ipynb             # Main analysis and modeling notebook
├── app.py               # Streamlit web application interface
├── voting_model.pkl     # Exported ensemble model artifact (joblib)
├── requirements.txt     # Python project dependencies
└── README.md            # Project documentation

```

---

## 🚀 Quick Start Guide

### 1. Prerequisites

Ensure **Python 3.8+** is installed on your environment.

### 2. Clone Repository & Setup Environment

```bash
# Clone repository
git clone [https://github.com/AICatalyst890/Student-Performance-Prediction-System.git](https://github.com/AICatalyst890/Student-Performance-Prediction-System.git)
cd Student-Performance-Prediction-System

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### 3. Run Analysis & Model Training (Optional)

To execute the data preprocessing pipeline, feature scaling, and model evaluation routines:

```bash
jupyter lab ml.ipynb

```

### 4. Launch Web Application

Run the following command to start the interactive Streamlit application:

```bash
streamlit run app.py

```

---

## 📄 License

This project is licensed under the [MIT License](https://github.com/AICatalyst890/Student-Performance-Prediction-System/blob/main/LICENSE).

```