"""
visualization.py - EDA & Results Visualization Module
=======================================================
All plotting functions for exploratory data analysis,
model evaluation charts, and result visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

# ── Global plot style ────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
FIGSIZE = (12, 7)
OUTPUT_DIR = "outputs"


def _save_fig(fig, filename):
    """Save figure to output directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    📊 Saved: {filepath}")


def plot_price_distribution(df):
    """Plot histogram + KDE of house prices."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram
    axes[0].hist(df["price"] / 100000, bins=50, color="#4C72B0", edgecolor="white", alpha=0.8)
    axes[0].set_title("House Price Distribution", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Price (₹ Lakhs)", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)

    # KDE
    sns.kdeplot(df["price"] / 100000, ax=axes[1], fill=True, color="#DD8452", alpha=0.6)
    axes[1].set_title("Price Density Plot", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Price (₹ Lakhs)", fontsize=12)

    fig.suptitle("🏠 House Price Distribution Analysis", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_fig(fig, "01_price_distribution.png")


def plot_correlation_heatmap(df):
    """Plot correlation heatmap for numerical features."""
    numerical_df = df.select_dtypes(include=[np.number])
    corr = numerical_df.corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt=".2f", square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax, annot_kws={"size": 8})

    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=20)
    fig.tight_layout()
    _save_fig(fig, "02_correlation_heatmap.png")


def plot_price_vs_area(df):
    """Scatter plot of price vs area colored by BHK."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    scatter = ax.scatter(df["area_sqft"], df["price"] / 100000,
                         c=df["bhk"], cmap="viridis", alpha=0.5, s=20, edgecolors="none")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("BHK", fontsize=12)
    ax.set_title("Price vs Area (by BHK)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Area (sq ft)", fontsize=12)
    ax.set_ylabel("Price (₹ Lakhs)", fontsize=12)
    fig.tight_layout()
    _save_fig(fig, "03_price_vs_area.png")


def plot_location_analysis(df):
    """Box plot of price by location."""
    fig, ax = plt.subplots(figsize=(14, 7))
    location_order = df.groupby("location")["price"].median().sort_values(ascending=False).index
    sns.boxplot(x="location", y=df["price"] / 100000, data=df, order=location_order,
                palette="coolwarm", ax=ax)
    ax.set_title("Price Distribution by Location", fontsize=14, fontweight="bold")
    ax.set_xlabel("Location", fontsize=12)
    ax.set_ylabel("Price (₹ Lakhs)", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    _save_fig(fig, "04_price_by_location.png")


def plot_bhk_analysis(df):
    """Bar plot of average price by BHK and count distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Average price by BHK
    bhk_price = df.groupby("bhk")["price"].mean() / 100000
    axes[0].bar(bhk_price.index, bhk_price.values, color=sns.color_palette("viridis", len(bhk_price)),
                edgecolor="white")
    axes[0].set_title("Average Price by BHK", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("BHK", fontsize=12)
    axes[0].set_ylabel("Avg Price (₹ Lakhs)", fontsize=12)

    # BHK distribution
    bhk_counts = df["bhk"].value_counts().sort_index()
    axes[1].bar(bhk_counts.index, bhk_counts.values, color=sns.color_palette("mako", len(bhk_counts)),
                edgecolor="white")
    axes[1].set_title("Property Count by BHK", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("BHK", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)

    fig.suptitle("🏠 BHK Analysis", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_fig(fig, "05_bhk_analysis.png")


def plot_furnishing_analysis(df):
    """Box plot and count for furnishing status."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.boxplot(x="furnishing", y=df["price"] / 100000, data=df,
                palette="Set2", ax=axes[0], order=["Unfurnished", "Semi-Furnished", "Furnished"])
    axes[0].set_title("Price by Furnishing Status", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Price (₹ Lakhs)", fontsize=12)

    furn_counts = df["furnishing"].value_counts()
    axes[1].pie(furn_counts.values, labels=furn_counts.index, autopct="%1.1f%%",
                colors=sns.color_palette("Set2"), startangle=90)
    axes[1].set_title("Furnishing Distribution", fontsize=14, fontweight="bold")

    fig.tight_layout()
    _save_fig(fig, "06_furnishing_analysis.png")


def plot_property_type_analysis(df):
    """Violin plot of price by property type."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.violinplot(x="property_type", y=df["price"] / 100000, data=df,
                   palette="muted", ax=ax, inner="box")
    ax.set_title("Price Distribution by Property Type", fontsize=14, fontweight="bold")
    ax.set_xlabel("Property Type", fontsize=12)
    ax.set_ylabel("Price (₹ Lakhs)", fontsize=12)
    fig.tight_layout()
    _save_fig(fig, "07_property_type_analysis.png")


def plot_age_vs_price(df):
    """Scatter plot of age vs price."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(df["age_years"], df["price"] / 100000, alpha=0.3, s=15, color="#E377C2")
    # Trend line
    z = np.polyfit(df["age_years"], df["price"] / 100000, 1)
    p = np.poly1d(z)
    x_line = np.linspace(df["age_years"].min(), df["age_years"].max(), 100)
    ax.plot(x_line, p(x_line), "--", color="red", linewidth=2, label=f"Trend (slope={z[0]:.1f})")
    ax.set_title("Property Age vs Price", fontsize=14, fontweight="bold")
    ax.set_xlabel("Age (Years)", fontsize=12)
    ax.set_ylabel("Price (₹ Lakhs)", fontsize=12)
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, "08_age_vs_price.png")


def plot_model_comparison(eval_df):
    """Bar chart comparing R² scores of all models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    models = eval_df["Model"].values
    r2_scores = eval_df["R2_raw"].values
    colors = sns.color_palette("viridis", len(models))

    # R² comparison
    bars = axes[0].barh(models, r2_scores, color=colors, edgecolor="white", height=0.6)
    axes[0].set_xlim(0, 1.05)
    axes[0].set_title("Model R² Score Comparison", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("R² Score", fontsize=12)
    for bar, score in zip(bars, r2_scores):
        axes[0].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{score:.4f}", va="center", fontsize=10)

    # MAE comparison
    mae_scores = eval_df["MAE_raw"].values / 100000
    bars2 = axes[1].barh(models, mae_scores, color=sns.color_palette("magma", len(models)),
                         edgecolor="white", height=0.6)
    axes[1].set_title("Model MAE Comparison", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("MAE (₹ Lakhs)", fontsize=12)
    for bar, score in zip(bars2, mae_scores):
        axes[1].text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                     f"{score:.2f}L", va="center", fontsize=10)

    fig.suptitle("🤖 Model Performance Comparison", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_fig(fig, "09_model_comparison.png")


def plot_actual_vs_predicted(y_test, y_pred, model_name="Best Model"):
    """Scatter plot of actual vs predicted prices."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    y_test_lakhs = np.array(y_test) / 100000
    y_pred_lakhs = np.array(y_pred) / 100000

    # Actual vs Predicted
    axes[0].scatter(y_test_lakhs, y_pred_lakhs, alpha=0.4, s=20, color="#4C72B0")
    max_val = max(y_test_lakhs.max(), y_pred_lakhs.max())
    axes[0].plot([0, max_val], [0, max_val], "--", color="red", linewidth=2, label="Perfect Prediction")
    axes[0].set_title(f"Actual vs Predicted ({model_name})", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Actual Price (₹ Lakhs)", fontsize=12)
    axes[0].set_ylabel("Predicted Price (₹ Lakhs)", fontsize=12)
    axes[0].legend()

    # Residuals
    residuals = y_test_lakhs - y_pred_lakhs
    axes[1].scatter(y_pred_lakhs, residuals, alpha=0.4, s=20, color="#55A868")
    axes[1].axhline(y=0, color="red", linestyle="--", linewidth=2)
    axes[1].set_title("Residual Plot", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Predicted Price (₹ Lakhs)", fontsize=12)
    axes[1].set_ylabel("Residual (₹ Lakhs)", fontsize=12)

    fig.suptitle(f"📈 {model_name} — Prediction Analysis", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_fig(fig, "10_actual_vs_predicted.png")


def plot_feature_importance(model, feature_cols, model_name="Model", top_n=15):
    """Plot feature importance for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        print(f"    ⚠ {model_name} doesn't support feature importance")
        return

    importance = model.feature_importances_
    indices = np.argsort(importance)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importance[indices],
            color=sns.color_palette("viridis", len(indices)), edgecolor="white")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_cols[i] for i in indices])
    ax.set_title(f"Top {top_n} Feature Importance ({model_name})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance", fontsize=12)
    fig.tight_layout()
    _save_fig(fig, "11_feature_importance.png")


def run_eda(df):
    """Run all EDA visualizations."""
    print("\n" + "─" * 60)
    print("  📊 EXPLORATORY DATA ANALYSIS")
    print("─" * 60)

    plot_price_distribution(df)
    plot_correlation_heatmap(df)
    plot_price_vs_area(df)
    plot_location_analysis(df)
    plot_bhk_analysis(df)
    plot_furnishing_analysis(df)
    plot_property_type_analysis(df)
    plot_age_vs_price(df)

    print(f"\n  ✓ EDA complete: 8 charts saved to {OUTPUT_DIR}/")
    print("─" * 60)


if __name__ == "__main__":
    df = pd.read_csv("data/housing_data.csv")
    run_eda(df)
