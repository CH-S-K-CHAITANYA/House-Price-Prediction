"""
model_evaluation.py - Model Evaluation & Comparison Module
============================================================
Calculates MAE, RMSE, R², Adjusted R² for all models and
generates comparison tables.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test, n_features=None):
    """
    Evaluate a single model on test data.
    
    Returns dict with MAE, RMSE, R², Adjusted R².
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Adjusted R²
    n = len(y_test)
    p = n_features if n_features else X_test.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted_R2": adj_r2,
        "y_pred": y_pred,
    }


def evaluate_all_models(results, X_test, y_test):
    """
    Evaluate all trained models and create a comparison table.
    
    Parameters
    ----------
    results : dict
        Output from model_training.train_models()
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.
    
    Returns
    -------
    tuple
        (evaluation_df, predictions_dict)
    """
    print("\n" + "─" * 60)
    print("  📈 MODEL EVALUATION")
    print("─" * 60)

    evaluations = []
    predictions = {}

    for name, data in results.items():
        model = data["model"]
        metrics = evaluate_model(model, X_test, y_test)
        predictions[name] = metrics["y_pred"]

        evaluations.append({
            "Model": name,
            "MAE (₹)": f"{metrics['MAE']:,.0f}",
            "RMSE (₹)": f"{metrics['RMSE']:,.0f}",
            "R² Score": f"{metrics['R2']:.4f}",
            "Adj. R²": f"{metrics['Adjusted_R2']:.4f}",
            "CV Mean R²": f"{data['cv_mean']:.4f}",
            "MAE_raw": metrics["MAE"],
            "RMSE_raw": metrics["RMSE"],
            "R2_raw": metrics["R2"],
        })

    eval_df = pd.DataFrame(evaluations).sort_values("R2_raw", ascending=False)

    print("\n  Model Comparison Table:")
    print("  " + "─" * 85)
    display_cols = ["Model", "MAE (₹)", "RMSE (₹)", "R² Score", "Adj. R²", "CV Mean R²"]
    print(eval_df[display_cols].to_string(index=False))
    print("  " + "─" * 85)

    # Identify best model
    best_idx = eval_df["R2_raw"].idxmax()
    best_model = eval_df.loc[best_idx, "Model"]
    best_r2 = eval_df.loc[best_idx, "R² Score"]
    print(f"\n  🏆 Best Performing Model: {best_model} (R² = {best_r2})")

    print("─" * 60)
    return eval_df, predictions


def print_detailed_evaluation(eval_df):
    """Print a formatted detailed evaluation report."""
    print("\n" + "=" * 60)
    print("  📊 DETAILED EVALUATION REPORT")
    print("=" * 60)

    for _, row in eval_df.iterrows():
        print(f"\n  📌 {row['Model']}")
        print(f"     MAE  : {row['MAE (₹)']}")
        print(f"     RMSE : {row['RMSE (₹)']}")
        print(f"     R²   : {row['R² Score']}")
        print(f"     Adj R²: {row['Adj. R²']}")
        print(f"     CV R² : {row['CV Mean R²']}")

        r2_val = row["R2_raw"]
        if r2_val >= 0.95:
            print("     Rating: ⭐⭐⭐⭐⭐ Excellent")
        elif r2_val >= 0.90:
            print("     Rating: ⭐⭐⭐⭐ Very Good")
        elif r2_val >= 0.80:
            print("     Rating: ⭐⭐⭐ Good")
        elif r2_val >= 0.70:
            print("     Rating: ⭐⭐ Fair")
        else:
            print("     Rating: ⭐ Needs Improvement")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("Run this module via main.py for full evaluation.")
