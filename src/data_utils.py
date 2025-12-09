"""
Data Utilities for Diabetes Prediction

Handles data loading, preprocessing, and validation for the
FHE-based machine learning pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_diabetes_data(
    path: str = "data/diabetes.csv",
    test_size: float = 0.2,
    random_state: int = 42,
    drop_columns: Optional[List[str]] = None,
    target_column: str = "Outcome",
    validate: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], StandardScaler]:
    """
    Load and preprocess the diabetes dataset.
    
    Args:
        path: Path to the CSV file
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        drop_columns: Columns to drop (e.g., ID columns)
        target_column: Name of the target variable
        validate: Whether to perform data validation
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names, scaler)
    """
    if drop_columns is None:
        drop_columns = ["Id", "id", "index", "INDEX"]
    
    logger.info(f"Loading data from {path}")
    
    # Load data
    df = pd.read_csv(path)
    original_shape = df.shape
    logger.info(f"Loaded dataset: {original_shape[0]} samples, {original_shape[1]} columns")
    
    # Drop ID-like columns
    cols_to_drop = [col for col in drop_columns if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"Dropped columns: {cols_to_drop}")
    
    # Validate data
    if validate:
        validation_report = validate_diabetes_data(df, target_column)
        if not validation_report["is_valid"]:
            logger.warning(f"Data validation issues: {validation_report['issues']}")
    
    # Separate features and target
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    feature_names = [col for col in df.columns if col != target_column]
    X = df[feature_names].values.astype(np.float64)
    y = df[target_column].values.astype(np.int32)
    
    logger.info(f"Features: {feature_names}")
    logger.info(f"Target distribution: {np.bincount(y)}")
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    
    logger.info("Applied StandardScaler normalization")
    
    return X_train_std, X_test_std, y_train, y_test, feature_names, scaler


def validate_diabetes_data(df: pd.DataFrame, target_column: str = "Outcome") -> Dict[str, Any]:
    """
    Validate the diabetes dataset for common issues.
    
    Args:
        df: DataFrame to validate
        target_column: Name of target column
        
    Returns:
        Validation report dictionary
    """
    issues = []
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        issues.append(f"Missing values: {missing[missing > 0].to_dict()}")
    
    # Check for unexpected zeros (physiologically impossible)
    zero_suspicious = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_suspicious:
        if col in df.columns:
            n_zeros = (df[col] == 0).sum()
            if n_zeros > 0:
                issues.append(f"{col} has {n_zeros} zero values (may be missing)")
    
    # Check target is binary
    if target_column in df.columns:
        unique_targets = df[target_column].unique()
        if not set(unique_targets).issubset({0, 1}):
            issues.append(f"Target has unexpected values: {unique_targets}")
    
    # Check for extreme outliers (more than 5 std from mean)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != target_column:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            n_outliers = (z_scores > 5).sum()
            if n_outliers > 0:
                issues.append(f"{col} has {n_outliers} extreme outliers (|z| > 5)")
    
    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "n_samples": len(df),
        "n_features": len(df.columns) - 1,
    }


def get_feature_statistics(
    X: np.ndarray,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Compute descriptive statistics for features.
    
    Args:
        X: Feature matrix
        feature_names: List of feature names
        
    Returns:
        DataFrame with statistics
    """
    stats = pd.DataFrame({
        "feature": feature_names,
        "mean": X.mean(axis=0),
        "std": X.std(axis=0),
        "min": X.min(axis=0),
        "max": X.max(axis=0),
        "median": np.median(X, axis=0),
    })
    return stats.round(4)


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "median",
    zero_as_missing: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        strategy: Imputation strategy ('mean', 'median', 'mode')
        zero_as_missing: Columns where 0 should be treated as missing
        
    Returns:
        DataFrame with imputed values
    """
    df = df.copy()
    
    # Treat zeros as missing for specified columns
    if zero_as_missing:
        for col in zero_as_missing:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)
    
    # Impute
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            if strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "mode":
                df[col] = df[col].fillna(df[col].mode()[0])
    
    return df


def create_sample_patient(
    feature_names: List[str],
    scaler: StandardScaler,
    profile: str = "average"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a sample patient for demonstration.
    
    Args:
        feature_names: List of feature names
        scaler: Fitted StandardScaler
        profile: Type of patient ('average', 'high_risk', 'low_risk')
        
    Returns:
        Tuple of (raw_features, standardized_features)
    """
    # Define profiles based on medical understanding
    profiles = {
        "average": {
            "Pregnancies": 3,
            "Glucose": 120,
            "BloodPressure": 70,
            "SkinThickness": 20,
            "Insulin": 80,
            "BMI": 25,
            "DiabetesPedigreeFunction": 0.5,
            "Age": 35,
        },
        "high_risk": {
            "Pregnancies": 6,
            "Glucose": 180,
            "BloodPressure": 85,
            "SkinThickness": 35,
            "Insulin": 200,
            "BMI": 35,
            "DiabetesPedigreeFunction": 1.2,
            "Age": 55,
        },
        "low_risk": {
            "Pregnancies": 1,
            "Glucose": 85,
            "BloodPressure": 65,
            "SkinThickness": 15,
            "Insulin": 50,
            "BMI": 22,
            "DiabetesPedigreeFunction": 0.2,
            "Age": 25,
        },
    }
    
    profile_values = profiles.get(profile, profiles["average"])
    raw = np.array([profile_values.get(f, 0) for f in feature_names], dtype=np.float64)
    standardized = scaler.transform(raw.reshape(1, -1)).flatten()
    
    return raw, standardized


def describe_dataset(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str]
) -> str:
    """
    Generate a text description of the dataset.
    
    Returns:
        Formatted string with dataset description
    """
    train_pos = y_train.sum()
    train_neg = len(y_train) - train_pos
    test_pos = y_test.sum()
    test_neg = len(y_test) - test_pos
    
    description = f"""
Dataset Summary
===============
Training samples: {len(X_train)} (Positive: {train_pos}, Negative: {train_neg})
Test samples:     {len(X_test)} (Positive: {test_pos}, Negative: {test_neg})
Features:         {len(feature_names)}

Feature names: {', '.join(feature_names)}

Class balance:
  Training: {train_pos/len(y_train):.1%} positive
  Test:     {test_pos/len(y_test):.1%} positive
"""
    return description