"""
model_training.py - Regression Model Training Module
======================================================
Trains 7 regression models with cross-validation and saves them.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")


def split_data(df, feature_cols, target_col="price", test_size=0.2, seed=42):
    """Split data into training and testing sets."""
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    print(f"  ✓ Data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def get_models():
    """Return dictionary of regression models to train."""
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Lasso Regression": Lasso(alpha=1000, random_state=42, max_iter=10000),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, verbosity=0, n_jobs=-1
        ),
    }
    return models


def train_models(X_train, y_train, X_test, y_test):
    """
    Train all regression models and perform cross-validation.
    
    Returns
    -------
    dict
        {model_name: {"model": fitted_model, "cv_scores": array, "cv_mean": float}}
    """
    models = get_models()
    results = {}

    print("\n" + "─" * 60)
    print("  🤖 MODEL TRAINING")
    print("─" * 60)

    for name, model in models.items():
        print(f"\n  Training: {name}...")

        # Train model
        model.fit(X_train, y_train)

        # Cross-validation on training set (5-fold)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

        results[name] = {
            "model": model,
            "cv_scores": cv_scores,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }

        print(f"    CV R² = {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    print("\n" + "─" * 60)
    return results


def save_models(results, save_dir="models"):
    """Save all trained models to disk using joblib."""
    os.makedirs(save_dir, exist_ok=True)
    for name, data in results.items():
        filename = name.lower().replace(" ", "_") + ".joblib"
        filepath = os.path.join(save_dir, filename)
        joblib.dump(data["model"], filepath)
    print(f"  ✓ {len(results)} models saved to {save_dir}/")


def load_model(model_name, models_dir="models"):
    """Load a saved model from disk."""
    filename = model_name.lower().replace(" ", "_") + ".joblib"
    filepath = os.path.join(models_dir, filename)
    model = joblib.load(filepath)
    print(f"  ✓ Loaded model: {model_name}")
    return model


def get_best_model(results):
    """Return the name and model with the highest CV R² score."""
    best_name = max(results, key=lambda k: results[k]["cv_mean"])
    best_model = results[best_name]["model"]
    best_score = results[best_name]["cv_mean"]
    print(f"\n  🏆 Best Model: {best_name} (CV R² = {best_score:.4f})")
    return best_name, best_model


if __name__ == "__main__":
    from data_preprocessing import preprocess_pipeline
    from feature_engineering import prepare_features

    df, encoders = preprocess_pipeline()
    df, feature_cols = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(df, feature_cols)
    results = train_models(X_train, y_train, X_test, y_test)
    save_models(results)
    get_best_model(results)
