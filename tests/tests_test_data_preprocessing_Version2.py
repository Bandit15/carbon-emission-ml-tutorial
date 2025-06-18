"""
Unit tests for data_preprocessing.py
"""

import os
import pandas as pd
import numpy as np
import pytest

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from data_preprocessing import load_data, preprocess_features, train_test_split_processed

@pytest.fixture
def sample_df():
    # Minimal sample data for testing
    return pd.DataFrame({
        "country": ["A", "B"],
        "year": [2020, 2021],
        "gdp_per_capita": [1000, 2000],
        "energy_consumption": [10, 20],
        "population": [1_000_000, 2_000_000],
        "renewable_energy_pct": [15.0, 30.0],
        "urban_pct": [50.0, 60.0],
        "co2_emission_per_capita": [1.5, 2.5],
        "reference_source": ["source1", "source2"]
    })

def test_load_data(tmp_path, sample_df):
    test_file = tmp_path / "test.csv"
    sample_df.to_csv(test_file, index=False)
    df_loaded = load_data(test_file)
    assert isinstance(df_loaded, pd.DataFrame)
    assert df_loaded.equals(sample_df)

def test_preprocess_features(sample_df):
    X_processed, y, preprocessor, feature_names = preprocess_features(sample_df)
    # Should encode 1 categorical feature (country) with 2 categories, drop='first' makes 1 dummy
    # 6 numeric + 1 dummy = 7 features
    assert X_processed.shape[1] == 7
    assert len(y) == 2
    assert "country_B" in feature_names

def test_train_test_split_processed(sample_df):
    X_train, X_test, y_train, y_test, preprocessor, feature_names = train_test_split_processed(
        sample_df, test_size=0.5, random_state=1
    )
    assert X_train.shape[0] == 1
    assert X_test.shape[0] == 1
    assert len(y_train) == 1
    assert len(y_test) == 1