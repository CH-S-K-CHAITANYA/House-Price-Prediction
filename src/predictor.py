"""
predictor.py - House Price Prediction Module
==============================================
Provides prediction interface for new property inputs
using the best trained model.
"""

import numpy as np
import pandas as pd
import joblib


def predict_price(model, input_data, feature_cols, scaler=None):
    """
    Predict house price for given property features.
    
    Parameters
    ----------
    model : sklearn estimator
        Trained regression model.
    input_data : dict
        Property features as key-value pairs.
    feature_cols : list
        List of feature column names (in order).
    scaler : StandardScaler, optional
        Fitted scaler for feature scaling.
    
    Returns
    -------
    float
        Predicted price in ₹.
    """
    # Create DataFrame from input
    input_df = pd.DataFrame([input_data])

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder to match training order
    input_df = input_df[feature_cols]

    # Scale if scaler provided
    if scaler is not None:
        input_df = pd.DataFrame(
            scaler.transform(input_df),
            columns=feature_cols
        )

    # Predict
    prediction = model.predict(input_df)[0]
    return max(0, prediction)


def format_price(price):
    """Format price in Indian numbering system (Lakhs/Crores)."""
    if price >= 10000000:
        return f"₹{price / 10000000:.2f} Crores"
    elif price >= 100000:
        return f"₹{price / 100000:.2f} Lakhs"
    else:
        return f"₹{price:,.0f}"


def demo_predictions(model, feature_cols, encoders=None, scaler=None):
    """Run demo predictions with sample property inputs."""
    print("\n" + "=" * 60)
    print("  💰 SAMPLE HOUSE PRICE PREDICTIONS")
    print("=" * 60)

    # Sample properties for demonstration
    samples = [
        {
            "desc": "2 BHK Apartment in Electronic City (New)",
            "data": {
                "area_sqft": 1100, "bhk": 2, "bathrooms": 2, "balconies": 1,
                "parking": 1, "age_years": 1, "floor": 5, "total_floors": 15,
                "property_type_encoded": 0, "furnishing_encoded": 1,
                "availability_encoded": 0, "location_encoded": 3,
                "amenities_score": 7.0, "distance_to_center_km": 22.0,
                "nearby_schools": 3, "nearby_hospitals": 2,
                "total_rooms": 5, "area_per_room": 220.0, "floor_ratio": 0.333,
                "is_top_floor": 0, "age_category_encoded": 5,
                "is_premium_location": 0, "luxury_index": 5.6,
                "bhk_area_interaction": 2200,
            }
        },
        {
            "desc": "3 BHK Furnished in Koramangala (5 yrs old)",
            "data": {
                "area_sqft": 1600, "bhk": 3, "bathrooms": 3, "balconies": 2,
                "parking": 2, "age_years": 5, "floor": 8, "total_floors": 12,
                "property_type_encoded": 0, "furnishing_encoded": 0,
                "availability_encoded": 0, "location_encoded": 5,
                "amenities_score": 8.5, "distance_to_center_km": 4.0,
                "nearby_schools": 4, "nearby_hospitals": 3,
                "total_rooms": 8, "area_per_room": 200.0, "floor_ratio": 0.667,
                "is_top_floor": 0, "age_category_encoded": 4,
                "is_premium_location": 1, "luxury_index": 9.05,
                "bhk_area_interaction": 4800,
            }
        },
        {
            "desc": "4 BHK Villa in Whitefield (Brand New)",
            "data": {
                "area_sqft": 2500, "bhk": 4, "bathrooms": 4, "balconies": 3,
                "parking": 2, "age_years": 0, "floor": 1, "total_floors": 3,
                "property_type_encoded": 2, "furnishing_encoded": 0,
                "availability_encoded": 0, "location_encoded": 10,
                "amenities_score": 9.0, "distance_to_center_km": 20.0,
                "nearby_schools": 5, "nearby_hospitals": 2,
                "total_rooms": 11, "area_per_room": 227.3, "floor_ratio": 0.333,
                "is_top_floor": 0, "age_category_encoded": 5,
                "is_premium_location": 0, "luxury_index": 9.2,
                "bhk_area_interaction": 10000,
            }
        },
    ]

    for i, sample in enumerate(samples, 1):
        price = predict_price(model, sample["data"], feature_cols, scaler)
        print(f"\n  🏠 Property {i}: {sample['desc']}")
        print(f"     💰 Predicted Price: {format_price(price)}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("Run via main.py for full prediction demo.")
