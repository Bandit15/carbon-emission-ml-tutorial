"""
Train Ridge and Lasso Regression Models for CO2 Emission Prediction

- Loads and preprocesses the dataset using data_preprocessing.py
- Trains Ridge and Lasso regression models
- Evaluates their performance on the test set
- Saves the trained models for reuse

Usage:
    python src/train_ridge_lasso.py
"""

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

from data_preprocessing import load_data, train_test_split_processed

def main():
    # Paths
    DATA_PATH = "data/carbon_emission_ml_dataset.csv"
    MODEL_DIR = "models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load and preprocess data
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, preprocessor, feature_names = train_test_split_processed(df)

    # Ridge Regression
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
    ridge_r2 = r2_score(y_test, ridge_pred)

    # Save Ridge model
    joblib.dump(ridge, os.path.join(MODEL_DIR, "ridge_model.pkl"))

    print("Ridge Regression Results:")
    print(f"  RMSE: {ridge_rmse:.3f}")
    print(f"  R2:   {ridge_r2:.3f}")

    # Lasso Regression
    lasso = Lasso(alpha=0.1, random_state=42)
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)
    lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))
    lasso_r2 = r2_score(y_test, lasso_pred)

    # Save Lasso model
    joblib.dump(lasso, os.path.join(MODEL_DIR, "lasso_model.pkl"))

    print("\nLasso Regression Results:")
    print(f"  RMSE: {lasso_rmse:.3f}")
    print(f"  R2:   {lasso_r2:.3f}")

    # Save preprocessor for deployment
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.pkl"))

if __name__ == "__main__":
    main()