"""
Data Preprocessing Module for Carbon Emission ML Tutorial

This script provides reusable functions for:
- Loading the carbon emission dataset
- Handling missing values
- Encoding categorical variables
- Feature scaling
- Splitting into train/test sets

Usage:
    from src.data_preprocessing import load_and_preprocess_data
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(filepath):
    """
    Loads the carbon emission dataset from the given CSV file path.

    Args:
        filepath (str): Path to the dataset CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    return pd.read_csv(filepath)


def preprocess_features(df, target_column="co2_emission_per_capita"):
    """
    Preprocesses the features:
    - Encodes categorical features (country)
    - Scales numerical features
    - Returns processed X, y, and the fitted pipeline

    Args:
        df (pd.DataFrame): The raw dataframe.
        target_column (str): Name of the target column.

    Returns:
        X_processed (np.ndarray): Processed feature array
        y (np.ndarray): Target array
        pipeline (Pipeline): Fitted preprocessing pipeline
        feature_names (list): Names after transformation
    """
    # Features and target
    X = df.drop([target_column, "reference_source"], axis=1)
    y = df[target_column].values

    categorical_features = ["country"]
    numeric_features = [col for col in X.columns if col not in categorical_features]

    # Define transformations
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Fit and transform
    X_processed = preprocessor.fit_transform(X)

    # Get feature names after transformation
    feature_names = (
        numeric_features +
        list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
    )

    return X_processed, y, preprocessor, feature_names


def train_test_split_processed(df, target_column="co2_emission_per_capita", test_size=0.2, random_state=42):
    """
    Full pipeline from raw dataframe to split and processed data.

    Args:
        df (pd.DataFrame): Raw dataframe.
        target_column (str): Target column name.
        test_size (float): Fraction for test set.
        random_state (int): Random seed.

    Returns:
        X_train, X_test, y_train, y_test, preprocessor, feature_names
    """
    X_processed, y, preprocessor, feature_names = preprocess_features(df, target_column)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, preprocessor, feature_names

# Example usage (uncomment for script use):
# if __name__ == "__main__":
#     df = load_data("../data/carbon_emission_ml_dataset.csv")
#     X_train, X_test, y_train, y_test, preprocessor, feature_names = train_test_split_processed(df)