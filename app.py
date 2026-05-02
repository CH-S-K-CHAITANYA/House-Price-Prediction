"""
app.py - Streamlit Interactive Dashboard
==========================================
Modern, industry-ready UI for house price prediction.
Features:
  - Interactive price prediction form
  - EDA visualizations with Plotly
  - Model comparison dashboard
  - Dataset explorer
  
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ── Page Configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(10px);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.3);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.6);
        margin-top: 4px;
    }

    .price-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 32px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
    }
    .price-result h2 {
        color: white;
        font-size: 2.4rem;
        margin: 0;
    }
    .price-result p {
        color: rgba(255,255,255,0.8);
        font-size: 1rem;
    }

    .sidebar .sidebar-content { background: rgba(15, 12, 41, 0.95); }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    h1, h2, h3 { color: white !important; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        color: white;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ─────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    """Load the housing dataset."""
    raw_path = "data/housing_data.csv"
    clean_path = "data/housing_data_cleaned.csv"
    df_raw = pd.read_csv(raw_path) if os.path.exists(raw_path) else None
    df_clean = pd.read_csv(clean_path) if os.path.exists(clean_path) else None
    return df_raw, df_clean


@st.cache_resource
def load_models():
    """Load all saved models."""
    models_dir = "models"
    models = {}
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith(".joblib"):
                name = f.replace(".joblib", "").replace("_", " ").title()
                models[name] = joblib.load(os.path.join(models_dir, f))
    return models


def format_inr(price):
    """Format price in Indian numbering."""
    if price >= 10000000:
        return f"₹{price / 10000000:.2f} Cr"
    elif price >= 100000:
        return f"₹{price / 100000:.2f} L"
    return f"₹{price:,.0f}"


# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 Navigation")
    page = st.radio(
        "Select Page",
        ["🏠 Home", "📊 EDA Dashboard", "🤖 Model Comparison", "💰 Price Predictor", "📁 Dataset Explorer"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### 🛠️ Project Info")
    st.markdown("""
    **Tech Stack:**
    - Python 3.10+
    - Scikit-learn & XGBoost
    - Streamlit & Plotly
    
    **Models:** 7 Regression Models  
    **Dataset:** 5,000 Synthetic Records
    """)
    st.markdown("---")
    st.markdown("Made with ❤️ for ML Portfolio")


# ── Load Data ────────────────────────────────────────────────────────
df_raw, df_clean = load_dataset()
models = load_models()

if df_raw is None:
    st.warning("⚠️ Dataset not found. Run `python main.py` first to generate data and train models.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("# 🏠 House Price Prediction System")
    st.markdown("### *End-to-End ML Pipeline with Interactive Dashboard*")
    st.markdown("---")

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{len(df_raw):,}</div>
            <div class="metric-label">Total Properties</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{df_raw.shape[1]}</div>
            <div class="metric-label">Features</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{len(models)}</div>
            <div class="metric-label">Trained Models</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        avg_price = format_inr(df_raw["price"].mean())
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{avg_price}</div>
            <div class="metric-label">Avg. Price</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick stats
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df_raw, x=df_raw["price"] / 100000, nbins=50,
                           title="Price Distribution (₹ Lakhs)",
                           color_discrete_sequence=["#667eea"])
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Price (₹ Lakhs)",
            yaxis_title="Count", font=dict(family="Inter")
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        loc_avg = df_raw.groupby("location")["price"].mean().sort_values(ascending=True) / 100000
        fig = px.bar(x=loc_avg.values, y=loc_avg.index, orientation="h",
                     title="Average Price by Location (₹ Lakhs)",
                     color=loc_avg.values, color_continuous_scale="viridis")
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Avg Price (₹ Lakhs)",
            yaxis_title="", font=dict(family="Inter"), showlegend=False
        )
        st.plotly_chart(fig, width="stretch")


# ══════════════════════════════════════════════════════════════════════
# PAGE: EDA DASHBOARD
# ══════════════════════════════════════════════════════════════════════
elif page == "📊 EDA Dashboard":
    st.markdown("# 📊 Exploratory Data Analysis")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🔗 Correlations", "📍 Location", "🏗️ Property"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(df_raw, x="area_sqft", y=df_raw["price"] / 100000,
                             color="bhk", title="Price vs Area by BHK",
                             color_continuous_scale="viridis", opacity=0.5)
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", yaxis_title="Price (₹ Lakhs)")
            st.plotly_chart(fig, width="stretch")
        with col2:
            bhk_avg = df_raw.groupby("bhk")["price"].mean() / 100000
            fig = px.bar(x=bhk_avg.index, y=bhk_avg.values, title="Avg Price by BHK",
                         color=bhk_avg.values, color_continuous_scale="plasma")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", yaxis_title="Avg Price (₹ Lakhs)",
                              xaxis_title="BHK")
            st.plotly_chart(fig, width="stretch")

    with tab2:
        numerical = df_raw.select_dtypes(include=[np.number])
        corr = numerical.corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        title="Feature Correlation Matrix", aspect="auto")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          width=800, height=700)
        st.plotly_chart(fig, width="stretch")

    with tab3:
        fig = px.box(df_raw, x="location", y=df_raw["price"] / 100000,
                     color="location", title="Price Range by Location")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                          yaxis_title="Price (₹ Lakhs)", xaxis_tickangle=-45)
        st.plotly_chart(fig, width="stretch")

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.violin(df_raw, x="property_type", y=df_raw["price"] / 100000,
                            color="property_type", title="Price by Property Type", box=True)
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                              yaxis_title="Price (₹ Lakhs)")
            st.plotly_chart(fig, width="stretch")
        with col2:
            fig = px.box(df_raw, x="furnishing", y=df_raw["price"] / 100000,
                         color="furnishing", title="Price by Furnishing")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                              yaxis_title="Price (₹ Lakhs)")
            st.plotly_chart(fig, width="stretch")


# ══════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.markdown("# 🤖 Model Performance Comparison")
    st.markdown("---")

    if not models:
        st.warning("⚠️ No trained models found. Run `python main.py` first.")
        st.stop()

    # Evaluate models on test set
    from src.feature_engineering import create_features, get_feature_columns
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    if df_clean is not None:
        df_eval = create_features(df_clean)
        feature_cols = get_feature_columns(df_eval)
        X = df_eval[feature_cols]
        y = df_eval["price"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        eval_data = []
        for name, model in models.items():
            try:
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                eval_data.append({"Model": name, "R²": r2, "MAE": mae, "RMSE": rmse})
            except Exception:
                pass

        if eval_data:
            eval_df = pd.DataFrame(eval_data).sort_values("R²", ascending=False)

            # R² Chart
            fig = px.bar(eval_df, x="R²", y="Model", orientation="h",
                         title="R² Score Comparison",
                         color="R²", color_continuous_scale="viridis")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, width="stretch")

            # Table
            display_df = eval_df.copy()
            display_df["MAE (₹)"] = display_df["MAE"].apply(lambda x: format_inr(x))
            display_df["RMSE (₹)"] = display_df["RMSE"].apply(lambda x: format_inr(x))
            display_df["R² Score"] = display_df["R²"].apply(lambda x: f"{x:.4f}")
            st.dataframe(display_df[["Model", "R² Score", "MAE (₹)", "RMSE (₹)"]],
                         width="stretch", hide_index=True)

            # Best model actual vs predicted
            best_name = eval_df.iloc[0]["Model"]
            best_model = models[best_name]
            y_pred_best = best_model.predict(X_test)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test / 100000, y=y_pred_best / 100000, mode="markers",
                marker=dict(color="#667eea", size=5, opacity=0.5), name="Predictions"
            ))
            max_v = max(y_test.max(), y_pred_best.max()) / 100000
            fig.add_trace(go.Scatter(
                x=[0, max_v], y=[0, max_v], mode="lines",
                line=dict(color="red", dash="dash"), name="Perfect"
            ))
            fig.update_layout(
                title=f"Actual vs Predicted — {best_name}",
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Actual Price (₹ Lakhs)",
                yaxis_title="Predicted Price (₹ Lakhs)",
            )
            st.plotly_chart(fig, width="stretch")


# ══════════════════════════════════════════════════════════════════════
# PAGE: PRICE PREDICTOR
# ══════════════════════════════════════════════════════════════════════
elif page == "💰 Price Predictor":
    st.markdown("# 💰 Predict House Price")
    st.markdown("### Enter property details below")
    st.markdown("---")

    if not models:
        st.warning("⚠️ No trained models found. Run `python main.py` first.")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        area = st.number_input("📐 Area (sq ft)", min_value=300, max_value=10000, value=1200, step=50)
        bhk = st.selectbox("🛏️ BHK", [1, 2, 3, 4, 5], index=1)
        bathrooms = st.selectbox("🚿 Bathrooms", [1, 2, 3, 4, 5], index=1)
        balconies = st.selectbox("🌅 Balconies", [0, 1, 2, 3], index=1)

    with col2:
        location = st.selectbox("📍 Location", sorted(df_raw["location"].unique()))
        property_type = st.selectbox("🏗️ Property Type", ["Apartment", "Villa", "Independent House"])
        furnishing = st.selectbox("🛋️ Furnishing", ["Unfurnished", "Semi-Furnished", "Furnished"])
        availability = st.selectbox("📅 Availability", ["Ready to Move", "Under Construction"])

    with col3:
        age = st.slider("📅 Age (years)", 0, 50, 5)
        parking = st.selectbox("🅿️ Parking", [0, 1, 2], index=1)
        floor_num = st.slider("🏢 Floor Number", 1, 25, 5)
        total_floors = st.slider("🏢 Total Floors", 1, 30, 15)
        amenities = st.slider("⭐ Amenities Score", 0.0, 10.0, 7.0, 0.5)

    model_choice = st.selectbox("🤖 Select Model", list(models.keys()))

    if st.button("🔮 Predict Price", width="stretch"):
        # Build encodings (matching training data)
        loc_map = {loc: i for i, loc in enumerate(sorted(df_raw["location"].unique()))}
        prop_map = {"Apartment": 0, "Independent House": 1, "Villa": 2}
        furn_map = {"Furnished": 0, "Semi-Furnished": 1, "Unfurnished": 2}
        avail_map = {"Ready to Move": 0, "Under Construction": 1}

        total_rooms = bhk + bathrooms + balconies
        furn_score = {"Unfurnished": 0, "Semi-Furnished": 1, "Furnished": 2}
        age_cat_map = {range(0, 3): 5, range(3, 6): 4, range(6, 11): 3,
                       range(11, 21): 2, range(21, 101): 1}
        age_cat = 3
        for r, v in age_cat_map.items():
            if age in r:
                age_cat = v
                break

        premium_locs = ["Koramangala", "Indiranagar", "HSR Layout", "Jayanagar"]

        features = {
            "area_sqft": area, "bhk": bhk, "bathrooms": bathrooms,
            "balconies": balconies, "parking": parking, "age_years": age,
            "floor": floor_num, "total_floors": total_floors,
            "property_type_encoded": prop_map.get(property_type, 0),
            "furnishing_encoded": furn_map.get(furnishing, 1),
            "availability_encoded": avail_map.get(availability, 0),
            "location_encoded": loc_map.get(location, 0),
            "amenities_score": amenities,
            "distance_to_center_km": 10.0,
            "nearby_schools": 3, "nearby_hospitals": 2,
            "total_rooms": total_rooms,
            "area_per_room": round(area / max(total_rooms, 1), 1),
            "floor_ratio": round(floor_num / max(total_floors, 1), 3),
            "is_top_floor": int(floor_num == total_floors),
            "age_category_encoded": age_cat,
            "is_premium_location": int(location in premium_locs),
            "luxury_index": round(amenities * 0.3 + parking * 2.0 + furn_score[furnishing] * 1.5 + balconies * 0.5, 2),
            "bhk_area_interaction": bhk * area,
        }

        from src.feature_engineering import get_feature_columns, create_features
        if df_clean is not None:
            df_temp = create_features(df_clean.head(1).copy())
            feature_cols = get_feature_columns(df_temp)
        else:
            feature_cols = list(features.keys())

        input_df = pd.DataFrame([features])
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_cols]

        model = models[model_choice]
        prediction = model.predict(input_df)[0]
        prediction = max(500000, prediction)

        st.markdown(f"""
        <div class="price-result">
            <p>Estimated Price</p>
            <h2>{format_inr(prediction)}</h2>
            <p>{bhk} BHK {property_type} • {area} sqft • {location}</p>
            <p>Model: {model_choice}</p>
        </div>
        """, unsafe_allow_html=True)

        # Price breakdown
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Price per sq ft", f"₹{prediction / area:,.0f}")
        with col2:
            st.metric("Monthly EMI (20yr @8.5%)", f"₹{prediction * 0.0087:,.0f}")
        with col3:
            st.metric("Down Payment (20%)", format_inr(prediction * 0.2))


# ══════════════════════════════════════════════════════════════════════
# PAGE: DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════
elif page == "📁 Dataset Explorer":
    st.markdown("# 📁 Dataset Explorer")
    st.markdown("---")

    tab1, tab2 = st.tabs(["📋 Raw Data", "🧹 Cleaned Data"])

    with tab1:
        st.markdown(f"**Shape:** {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
        st.dataframe(df_raw.head(100), width="stretch", height=400)
        st.markdown("### 📊 Statistics")
        st.dataframe(df_raw.describe().round(2), width="stretch")

    with tab2:
        if df_clean is not None:
            st.markdown(f"**Shape:** {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
            st.dataframe(df_clean.head(100), width="stretch", height=400)
        else:
            st.info("Cleaned dataset not available. Run `python main.py` first.")
