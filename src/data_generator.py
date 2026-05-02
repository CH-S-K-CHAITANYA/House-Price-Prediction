"""
data_generator.py - Synthetic Housing Dataset Generator
========================================================
Generates a realistic synthetic housing dataset with 5000 records.
Features are designed to mimic real Indian real estate market patterns,
with realistic correlations between features and price.
"""

import numpy as np
import pandas as pd
import os

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
NUM_RECORDS = 5000
MISSING_FRACTION = 0.02  # 2% missing values for realism

# Location data with base price multipliers (₹ per sqft)
LOCATIONS = {
    "Whitefield":      {"base_rate": 6500, "weight": 0.12},
    "Koramangala":     {"base_rate": 12000, "weight": 0.08},
    "HSR Layout":      {"base_rate": 9500, "weight": 0.10},
    "Electronic City": {"base_rate": 5000, "weight": 0.12},
    "Indiranagar":     {"base_rate": 13000, "weight": 0.07},
    "Jayanagar":       {"base_rate": 10000, "weight": 0.08},
    "Marathahalli":    {"base_rate": 6000, "weight": 0.11},
    "Sarjapur Road":   {"base_rate": 5500, "weight": 0.10},
    "Hebbal":          {"base_rate": 7500, "weight": 0.09},
    "Yelahanka":       {"base_rate": 4500, "weight": 0.08},
    "Banashankari":    {"base_rate": 7000, "weight": 0.05},
}

FURNISHING_OPTIONS = ["Unfurnished", "Semi-Furnished", "Furnished"]
FURNISHING_MULTIPLIER = {"Unfurnished": 0.90, "Semi-Furnished": 1.00, "Furnished": 1.15}

PROPERTY_TYPES = ["Apartment", "Villa", "Independent House"]
PROPERTY_MULTIPLIER = {"Apartment": 1.0, "Villa": 1.35, "Independent House": 1.20}

AVAILABILITY = ["Ready to Move", "Under Construction"]
AVAILABILITY_MULTIPLIER = {"Ready to Move": 1.05, "Under Construction": 0.92}


def generate_housing_data(n_records=NUM_RECORDS, seed=RANDOM_SEED):
    """
    Generate synthetic housing data with realistic feature correlations.

    Parameters
    ----------
    n_records : int
        Number of records to generate (default: 5000).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic housing dataset with 15 features and price target.
    """
    np.random.seed(seed)

    # ── Location ─────────────────────────────────────────────────────
    location_names = list(LOCATIONS.keys())
    location_weights = [LOCATIONS[loc]["weight"] for loc in location_names]
    locations = np.random.choice(location_names, size=n_records, p=location_weights)

    # ── Property Type ────────────────────────────────────────────────
    property_type = np.random.choice(
        PROPERTY_TYPES, size=n_records, p=[0.60, 0.15, 0.25]
    )

    # ── BHK (Bedrooms) ──────────────────────────────────────────────
    bhk = np.random.choice([1, 2, 3, 4, 5], size=n_records, p=[0.10, 0.35, 0.35, 0.15, 0.05])

    # ── Area (sqft) — correlated with BHK ────────────────────────────
    base_area = {1: 550, 2: 950, 3: 1400, 4: 2000, 5: 2800}
    area = np.array([
        max(300, int(np.random.normal(base_area[b], base_area[b] * 0.20)))
        for b in bhk
    ])

    # ── Bathrooms — correlated with BHK ─────────────────────────────
    bathrooms = np.array([
        min(b + np.random.choice([0, 1], p=[0.6, 0.4]), b + 1)
        for b in bhk
    ])

    # ── Balconies ────────────────────────────────────────────────────
    balconies = np.array([
        np.random.choice([0, 1, 2, 3], p=[0.15, 0.40, 0.35, 0.10])
        for _ in range(n_records)
    ])

    # ── Parking ──────────────────────────────────────────────────────
    parking = np.array([
        np.random.choice([0, 1, 2], p=[0.20, 0.55, 0.25])
        for _ in range(n_records)
    ])

    # ── Age of Property (years) ──────────────────────────────────────
    age = np.random.exponential(scale=8, size=n_records).astype(int)
    age = np.clip(age, 0, 50)

    # ── Floor Number ─────────────────────────────────────────────────
    total_floors = np.random.choice(
        [3, 5, 7, 10, 15, 20, 25], size=n_records,
        p=[0.10, 0.15, 0.20, 0.25, 0.15, 0.10, 0.05]
    )
    floor = np.array([
        np.random.randint(1, tf + 1) for tf in total_floors
    ])

    # ── Furnishing ───────────────────────────────────────────────────
    furnishing = np.random.choice(
        FURNISHING_OPTIONS, size=n_records, p=[0.30, 0.45, 0.25]
    )

    # ── Availability ─────────────────────────────────────────────────
    availability = np.random.choice(
        AVAILABILITY, size=n_records, p=[0.70, 0.30]
    )

    # ── Amenities Score (0-10) ───────────────────────────────────────
    # Higher in premium locations
    amenity_base = {loc: min(10, max(2, rate / 1500)) for loc, rate in
                    [(l, LOCATIONS[l]["base_rate"]) for l in location_names]}
    amenities_score = np.array([
        min(10, max(0, round(np.random.normal(amenity_base.get(loc, 5), 1.5), 1)))
        for loc in locations
    ])

    # ── Distance to City Center (km) ────────────────────────────────
    distance_base = {
        "Koramangala": 4, "Indiranagar": 5, "Jayanagar": 6,
        "HSR Layout": 10, "Banashankari": 8, "Marathahalli": 14,
        "Whitefield": 20, "Sarjapur Road": 18, "Hebbal": 12,
        "Electronic City": 22, "Yelahanka": 16,
    }
    distance_to_center = np.array([
        max(1, round(np.random.normal(distance_base.get(loc, 15), 3), 1))
        for loc in locations
    ])

    # ── Nearby Schools (count) ───────────────────────────────────────
    nearby_schools = np.random.poisson(lam=3, size=n_records)
    nearby_schools = np.clip(nearby_schools, 0, 10)

    # ── Nearby Hospitals (count) ─────────────────────────────────────
    nearby_hospitals = np.random.poisson(lam=2, size=n_records)
    nearby_hospitals = np.clip(nearby_hospitals, 0, 8)

    # ══════════════════════════════════════════════════════════════════
    # PRICE CALCULATION (Target Variable)
    # ══════════════════════════════════════════════════════════════════
    price = []
    for i in range(n_records):
        loc = locations[i]
        base_rate = LOCATIONS[loc]["base_rate"]

        # Base price = area × location rate
        p = area[i] * base_rate

        # BHK premium
        p *= (1 + (bhk[i] - 2) * 0.05)

        # Furnishing multiplier
        p *= FURNISHING_MULTIPLIER[furnishing[i]]

        # Property type multiplier
        p *= PROPERTY_MULTIPLIER[property_type[i]]

        # Availability multiplier
        p *= AVAILABILITY_MULTIPLIER[availability[i]]

        # Age depreciation (older = cheaper)
        p *= max(0.60, 1 - age[i] * 0.008)

        # Parking premium
        p *= (1 + parking[i] * 0.03)

        # Amenities premium
        p *= (1 + amenities_score[i] * 0.01)

        # Floor premium (higher floors slightly more expensive in apartments)
        if property_type[i] == "Apartment":
            p *= (1 + (floor[i] / total_floors[i]) * 0.05)

        # Random noise (±8%)
        noise = np.random.normal(1.0, 0.08)
        p *= noise

        # Round to nearest 10,000
        p = round(p / 10000) * 10000
        price.append(max(500000, p))  # Minimum ₹5 lakh

    # ── Build DataFrame ──────────────────────────────────────────────
    df = pd.DataFrame({
        "area_sqft": area,
        "bhk": bhk,
        "bathrooms": bathrooms,
        "balconies": balconies,
        "parking": parking,
        "age_years": age,
        "floor": floor,
        "total_floors": total_floors,
        "property_type": property_type,
        "furnishing": furnishing,
        "availability": availability,
        "location": locations,
        "amenities_score": amenities_score,
        "distance_to_center_km": distance_to_center,
        "nearby_schools": nearby_schools,
        "nearby_hospitals": nearby_hospitals,
        "price": price,
    })

    # ── Inject missing values for realism ────────────────────────────
    n_missing = int(n_records * MISSING_FRACTION)
    cols_for_missing = ["amenities_score", "balconies", "parking", "nearby_schools"]
    for col in cols_for_missing:
        missing_idx = np.random.choice(n_records, size=n_missing // len(cols_for_missing), replace=False)
        df.loc[missing_idx, col] = np.nan

    # ── Inject a few duplicate rows for realism ──────────────────────
    n_duplicates = 15
    dup_idx = np.random.choice(n_records, size=n_duplicates, replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

    return df


def save_dataset(df, filepath="data/housing_data.csv"):
    """Save the generated dataset to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"  ✓ Dataset saved: {filepath} ({df.shape[0]} rows × {df.shape[1]} columns)")
    return filepath


# ──────────────────────────────────────────────────────────────────────
# Main execution
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating synthetic housing dataset...")
    data = generate_housing_data()
    save_dataset(data)
    print("\nDataset Preview:")
    print(data.head(10))
    print(f"\nShape: {data.shape}")
    print(f"\nColumn Types:\n{data.dtypes}")
    print(f"\nMissing Values:\n{data.isnull().sum()}")
