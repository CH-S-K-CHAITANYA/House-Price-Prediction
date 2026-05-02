"""
main.py - House Price Prediction Pipeline
===========================================
Entry point that runs the complete ML pipeline:
  1. Generate synthetic housing dataset
  2. Preprocess and clean data
  3. Exploratory Data Analysis (EDA)
  4. Feature Engineering
  5. Model Training (7 regression models)
  6. Model Evaluation & Comparison
  7. Sample Price Predictions
  8. Visualization & Charts

Usage:
    python main.py

Author: House Price Prediction Project
"""

import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_generator import generate_housing_data, save_dataset
from src.data_preprocessing import preprocess_pipeline
from src.feature_engineering import prepare_features
from src.model_training import split_data, train_models, save_models, get_best_model
from src.model_evaluation import evaluate_all_models, print_detailed_evaluation
from src.visualization import (
    run_eda, plot_model_comparison, plot_actual_vs_predicted, plot_feature_importance
)
from src.predictor import demo_predictions


def print_banner():
    """Print the project banner."""
    print("\n")
    print("=" * 60)
    print("  🏠  HOUSE PRICE PREDICTION SYSTEM")
    print("  ─────────────────────────────────────")
    print("  Regression Models | ML Pipeline | EDA")
    print("=" * 60)
    print()


def main():
    """Run the complete House Price Prediction pipeline."""
    start_time = time.time()
    print_banner()

    # ── Phase 1: Generate Dataset ────────────────────────────────────
    print("[Phase 1] Generating synthetic housing dataset...")
    df_raw = generate_housing_data(n_records=5000)
    save_dataset(df_raw, "data/housing_data.csv")
    print(f"  Preview:\n{df_raw.head(5).to_string()}\n")

    # ── Phase 2: Preprocessing ───────────────────────────────────────
    print("\n[Phase 2] Cleaning and preprocessing data...")
    df_clean, encoders = preprocess_pipeline("data/housing_data.csv")

    # ── Phase 3: EDA ─────────────────────────────────────────────────
    print("\n[Phase 3] Performing Exploratory Data Analysis...")
    run_eda(df_raw)  # Use raw data for EDA visualizations

    # ── Phase 4: Feature Engineering ─────────────────────────────────
    print("\n[Phase 4] Engineering features...")
    df_featured, feature_cols = prepare_features(df_clean)

    # ── Phase 5: Train-Test Split ────────────────────────────────────
    print("\n[Phase 5] Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df_featured, feature_cols)

    # ── Phase 6: Model Training ──────────────────────────────────────
    print("\n[Phase 6] Training regression models...")
    results = train_models(X_train, y_train, X_test, y_test)

    # ── Phase 7: Model Evaluation ────────────────────────────────────
    print("\n[Phase 7] Evaluating models...")
    eval_df, predictions = evaluate_all_models(results, X_test, y_test)
    print_detailed_evaluation(eval_df)

    # ── Phase 8: Save Models ─────────────────────────────────────────
    print("\n[Phase 8] Saving models...")
    save_models(results)

    # ── Phase 9: Visualization ───────────────────────────────────────
    print("\n[Phase 9] Creating result visualizations...")

    # Model comparison chart
    plot_model_comparison(eval_df)

    # Best model analysis
    best_name, best_model = get_best_model(results)
    best_predictions = predictions[best_name]
    plot_actual_vs_predicted(y_test, best_predictions, best_name)
    plot_feature_importance(best_model, feature_cols, best_name)

    # ── Phase 10: Sample Predictions ─────────────────────────────────
    print("\n[Phase 10] Running sample predictions...")
    demo_predictions(best_model, feature_cols)

    # ── Complete ─────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  ✅ PIPELINE COMPLETE!")
    print(f"  ⏱  Total time: {elapsed:.1f} seconds")
    print(f"  📁 Outputs saved to: outputs/")
    print(f"  🤖 Models saved to: models/")
    print(f"  🏆 Best Model: {best_name}")
    print(f"  📊 Run 'streamlit run app.py' for interactive dashboard")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
