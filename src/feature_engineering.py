"""
feature_engineering.py - Feature Engineering Module
=====================================================
Creates derived features, interaction terms, and prepares
the final feature matrix for model training.
"""

import numpy as np
import pandas as pd


def create_features(df):
    """
    Create engineered features from existing columns.
    
    New features:
    - price_per_sqft: Price divided by area (only if training)
    - total_rooms: bhk + bathrooms + balconies
    - age_category: Categorical binning of property age
    - luxury_index: Composite score from amenities, parking, furnishing
    - area_per_room: Area divided by total rooms
    - floor_ratio: floor / total_floors
    - is_top_floor: Whether property is on the top floor
    - location_premium: Binary flag for premium locations
    """
    df = df.copy()

    # Total rooms = bedrooms + bathrooms + balconies
    df["total_rooms"] = df["bhk"] + df["bathrooms"] + df["balconies"]

    # Area per room
    df["area_per_room"] = (df["area_sqft"] / df["total_rooms"].replace(0, 1)).round(1)

    # Floor ratio (how high in the building)
    df["floor_ratio"] = (df["floor"] / df["total_floors"].replace(0, 1)).round(3)

    # Is top floor
    df["is_top_floor"] = (df["floor"] == df["total_floors"]).astype(int)

    # Age category
    df["age_category"] = pd.cut(
        df["age_years"],
        bins=[-1, 2, 5, 10, 20, 100],
        labels=["New", "Recent", "Moderate", "Old", "Very Old"]
    ).astype(str)

    # Age category encoded
    age_map = {"New": 5, "Recent": 4, "Moderate": 3, "Old": 2, "Very Old": 1}
    df["age_category_encoded"] = df["age_category"].map(age_map).fillna(3).astype(int)

    # Premium location flag
    premium_locations = ["Koramangala", "Indiranagar", "HSR Layout", "Jayanagar"]
    if "location" in df.columns:
        df["is_premium_location"] = df["location"].isin(premium_locations).astype(int)
    elif "location_encoded" in df.columns:
        df["is_premium_location"] = 0  # fallback

    # Luxury index (composite score)
    furnishing_score = 0
    if "furnishing_encoded" in df.columns:
        furnishing_score = df["furnishing_encoded"]
    elif "furnishing" in df.columns:
        furn_map = {"Unfurnished": 0, "Semi-Furnished": 1, "Furnished": 2}
        furnishing_score = df["furnishing"].map(furn_map).fillna(1)

    df["luxury_index"] = (
        df["amenities_score"] * 0.3 +
        df["parking"] * 2.0 +
        furnishing_score * 1.5 +
        df["balconies"] * 0.5
    ).round(2)

    # BHK-Area interaction
    df["bhk_area_interaction"] = (df["bhk"] * df["area_sqft"]).astype(int)

    print(f"  ✓ Feature engineering complete: {len(df.columns)} total columns")
    print(f"    New features: total_rooms, area_per_room, floor_ratio, is_top_floor,")
    print(f"    age_category_encoded, is_premium_location, luxury_index, bhk_area_interaction")

    return df


def get_feature_columns(df):
    """
    Return the list of feature columns to use for model training.
    Excludes target variable, raw categorical columns, and intermediate columns.
    """
    exclude_cols = [
        "price", "price_per_sqft",
        "property_type", "furnishing", "availability", "location",
        "age_category",
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    print(f"  ✓ Selected {len(feature_cols)} features for modeling")
    return feature_cols


def prepare_features(df):
    """Run feature engineering and return (df_engineered, feature_columns)."""
    print("\n" + "─" * 60)
    print("  ⚙️  FEATURE ENGINEERING")
    print("─" * 60)
    df = create_features(df)
    feature_cols = get_feature_columns(df)
    print("─" * 60)
    return df, feature_cols


if __name__ == "__main__":
    from data_preprocessing import preprocess_pipeline
    df, _ = preprocess_pipeline()
    df, feature_cols = prepare_features(df)
    print(f"\nFeature columns ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"  - {col}")
