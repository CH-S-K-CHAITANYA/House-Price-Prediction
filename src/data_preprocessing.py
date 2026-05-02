"""
data_preprocessing.py - Data Cleaning & Preprocessing Module
==============================================================
Handles missing values, duplicate removal, outlier treatment,
data type corrections, and encoding of categorical variables.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import warnings
warnings.filterwarnings("ignore")


def load_data(filepath="data/housing_data.csv"):
    """Load the housing dataset from CSV."""
    df = pd.read_csv(filepath)
    print(f"  ✓ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def data_summary(df):
    """Print a comprehensive summary of the dataset."""
    print("\n" + "=" * 60)
    print("  📊 DATASET SUMMARY")
    print("=" * 60)
    print(f"  Shape: {df.shape}")
    print(f"  Missing Values:\n{df.isnull().sum().to_string()}")
    print(f"  Duplicates: {df.duplicated().sum()}")
    print("=" * 60)


def handle_missing_values(df):
    """Fill missing values: median for numerical, mode for categorical."""
    missing_before = df.isnull().sum().sum()
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    print(f"  ✓ Missing values: {missing_before} → {df.isnull().sum().sum()}")
    return df


def remove_duplicates(df):
    """Remove duplicate rows."""
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  ✓ Duplicates removed: {n_before - len(df)} (rows: {n_before} → {len(df)})")
    return df


def treat_outliers(df, columns=None, factor=1.5):
    """Cap outliers using IQR method (winsorization)."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if "price" in columns:
            columns.remove("price")
    total_capped = 0
    for col in columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - factor * IQR, Q3 + factor * IQR
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_outliers > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            total_capped += n_outliers
    print(f"  ✓ Outliers capped (IQR): {total_capped} values")
    return df


def fix_data_types(df):
    """Ensure correct data types for all columns."""
    int_cols = ["bhk", "bathrooms", "balconies", "parking", "floor",
                "total_floors", "age_years", "nearby_schools", "nearby_hospitals"]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    print("  ✓ Data types corrected")
    return df


def encode_categorical(df):
    """Encode categorical variables using Label Encoding. Returns (df, encoders)."""
    categorical_cols = df.select_dtypes(include=["object"]).columns
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + "_encoded"] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"    → {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    print(f"  ✓ Encoded {len(categorical_cols)} categorical columns")
    return df, encoders


def scale_features(X_train, X_test, feature_columns):
    """Scale features using StandardScaler (fit on train only)."""
    scaler = StandardScaler()
    X_train_scaled, X_test_scaled = X_train.copy(), X_test.copy()
    X_train_scaled[feature_columns] = scaler.fit_transform(X_train[feature_columns])
    X_test_scaled[feature_columns] = scaler.transform(X_test[feature_columns])
    print(f"  ✓ Features scaled: {len(feature_columns)} columns")
    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(filepath="data/housing_data.csv", save_cleaned=True):
    """Run the complete preprocessing pipeline. Returns (cleaned_df, encoders)."""
    print("\n" + "─" * 60)
    print("  🔧 PREPROCESSING PIPELINE")
    print("─" * 60)
    df = load_data(filepath)
    data_summary(df)
    print("\n  [Step 1] Handling missing values...")
    df = handle_missing_values(df)
    print("\n  [Step 2] Removing duplicates...")
    df = remove_duplicates(df)
    print("\n  [Step 3] Treating outliers...")
    df = treat_outliers(df)
    print("\n  [Step 4] Fixing data types...")
    df = fix_data_types(df)
    print("\n  [Step 5] Encoding categorical variables...")
    df, encoders = encode_categorical(df)
    if save_cleaned:
        cleaned_path = filepath.replace(".csv", "_cleaned.csv")
        os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
        df.to_csv(cleaned_path, index=False)
        print(f"\n  ✓ Cleaned dataset saved: {cleaned_path}")
    print("─" * 60)
    return df, encoders


if __name__ == "__main__":
    df, encoders = preprocess_pipeline()
    print("\nCleaned Data Preview:")
    print(df.head())
