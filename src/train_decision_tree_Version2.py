"""
Train a Decision Tree Regressor for CO2 Emission Prediction

- Loads and preprocesses the dataset using data_preprocessing.py
- Trains a DecisionTreeRegressor model
- Evaluates its performance on the test set
- Saves the trained model and preprocessor for reuse

Usage:
    python src/train_decision_tree.py
"""

import os
import joblib
import numpy as np
from sklearn.tree import DecisionTreeRegressor
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

    # Train Decision Tree
    dtree = DecisionTreeRegressor(random_state=42)
    dtree.fit(X_train, y_train)
    y_pred = dtree.predict(X_test)

    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("Decision Tree Regression Results:")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  R2:   {r2:.3f}")

    # Save model and preprocessor
    joblib.dump(dtree, os.path.join(MODEL_DIR, "decision_tree_model.pkl"))
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.pkl"))

if __name__ == "__main__":
    main()