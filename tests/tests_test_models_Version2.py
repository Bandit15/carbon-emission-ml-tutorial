"""
Unit tests for model training scripts (basic fit & predict check)
"""

import os
import numpy as np
import pandas as pd
import pytest

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from data_preprocessing import preprocess_features

from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import lightgbm as lgb

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "country": ["A", "B", "C", "A"],
        "year": [2020, 2021, 2021, 2022],
        "gdp_per_capita": [1000, 2000, 1500, 1800],
        "energy_consumption": [10, 20, 15, 18],
        "population": [1_000_000, 2_000_000, 1_500_000, 1_800_000],
        "renewable_energy_pct": [15.0, 30.0, 22.0, 25.0],
        "urban_pct": [50.0, 60.0, 55.0, 58.0],
        "co2_emission_per_capita": [1.5, 2.5, 2.0, 2.2],
        "reference_source": ["s1", "s2", "s3", "s4"]
    })

@pytest.mark.parametrize("ModelClass,kwargs", [
    (Ridge, {"alpha": 1.0}),
    (Lasso, {"alpha": 0.1}),
    (DecisionTreeRegressor, {"random_state": 42}),
    (xgb.XGBRegressor, {"n_estimators": 10, "random_state": 42, "verbosity": 0}),
    (lgb.LGBMRegressor, {"n_estimators": 10, "random_state": 42}),
])
def test_model_fit_predict(ModelClass, kwargs, sample_df):
    X, y, preprocessor, feature_names = preprocess_features(sample_df)
    model = ModelClass(**kwargs)
    model.fit(X, y)
    y_pred = model.predict(X)
    # Check output shape and type
    assert y_pred.shape == y.shape
    assert isinstance(y_pred, np.ndarray)