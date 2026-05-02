# 🏠 House Price Prediction using Regression Models

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> End-to-end Machine Learning pipeline for predicting house prices using 7 regression models, complete with interactive Streamlit dashboard, comprehensive EDA, and feature engineering.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Industry Relevance](#-industry-relevance)
- [Tech Stack](#-tech-stack)
- [Project Architecture](#-project-architecture)
- [Dataset](#-dataset)
- [Models Used](#-models-used)
- [Results](#-results)
- [How to Run](#-how-to-run)
- [Screenshots](#-screenshots)
- [Folder Structure](#-folder-structure)
- [Learning Outcomes](#-learning-outcomes)

---

## 🎯 Overview

This project builds a **complete ML pipeline** for predicting residential property prices based on features like area, location, BHK, age, furnishing status, and more. It demonstrates industry-standard practices in data science — from data generation and cleaning to model training, evaluation, and deployment via an interactive dashboard.

**Key Highlights:**
- 🔢 **5,000+** synthetic property records with realistic distributions
- 🤖 **7 regression models** trained and compared
- 📊 **11+ visualizations** for EDA and model evaluation
- 🌐 **Interactive Streamlit dashboard** for real-time predictions
- 📈 **Best model achieves R² > 0.95** on test data

---

## ❓ Problem Statement

> Given a set of property features (area, BHK, location, age, furnishing, etc.), predict the market price of the house.

This is a **supervised regression problem** where we learn the mapping from property attributes to continuous price values.

---

## 🏢 Industry Relevance

| Stakeholder | Use Case |
|---|---|
| **Real Estate Portals** | Instant property valuation (like Zillow's Zestimate) |
| **Banks & Lenders** | Collateral assessment for home loans |
| **Investors** | Identify undervalued properties |
| **Buyers** | Know fair market value before purchasing |
| **Government** | Property tax assessment |
| **Brokers** | Data-driven competitive pricing |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy, SciPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| ML Models | Scikit-learn, XGBoost |
| Dashboard | Streamlit |
| Model Persistence | Joblib |

---

## 🏗️ Project Architecture

```
Raw Data → Preprocessing → Feature Engineering → Model Training → Evaluation → Prediction
    │           │                  │                    │              │            │
    ▼           ▼                  ▼                    ▼              ▼            ▼
 5000 rows   Clean data      8 new features      7 models      MAE/RMSE/R²   Dashboard
             Handle nulls    Luxury index         Cross-val     Comparison     Real-time
             Remove outliers Age categories       Grid search   Best model     EMI calc
```

---

## 📊 Dataset

**Type:** Synthetic (generated to mimic real Indian real estate market)

| Feature | Description |
|---|---|
| `area_sqft` | Built-up area in square feet |
| `bhk` | Number of bedrooms (1-5) |
| `bathrooms` | Number of bathrooms |
| `balconies` | Number of balconies |
| `parking` | Number of parking spaces |
| `age_years` | Age of the property |
| `floor` / `total_floors` | Floor number and building height |
| `property_type` | Apartment / Villa / Independent House |
| `furnishing` | Unfurnished / Semi-Furnished / Furnished |
| `location` | 11 Bangalore localities |
| `amenities_score` | Score (0-10) for nearby amenities |
| `distance_to_center_km` | Distance to city center |
| `nearby_schools` / `nearby_hospitals` | Count of nearby facilities |
| **`price`** | **Target variable (₹)** |

---

## 🤖 Models Used

| # | Model | Type |
|---|---|---|
| 1 | Linear Regression | Linear |
| 2 | Ridge Regression | Linear (L2 regularization) |
| 3 | Lasso Regression | Linear (L1 regularization) |
| 4 | Decision Tree Regressor | Tree-based |
| 5 | Random Forest Regressor | Ensemble (Bagging) |
| 6 | Gradient Boosting Regressor | Ensemble (Boosting) |
| 7 | XGBoost Regressor | Ensemble (Boosting) |

---

## 📈 Results

| Model | MAE (₹) | RMSE (₹) | R² Score |
|---|---|---|---|
| **Gradient Boosting** | **₹11.20L** | **₹19.97L** | **0.9506** |
| XGBoost | ₹11.58L | ₹20.99L | 0.9454 |
| Random Forest | ₹15.68L | ₹26.32L | 0.9142 |
| Linear Regression | ₹24.91L | ₹37.03L | 0.8301 |
| Ridge Regression | ₹24.90L | ₹37.03L | 0.8301 |
| Lasso Regression | ₹24.90L | ₹37.03L | 0.8301 |
| Decision Tree | ₹23.20L | ₹38.75L | 0.8140 |

> 🏆 **Best Model: Gradient Boosting** with R² = 0.9506

---

## 🚀 How to Run

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/House-Price-Prediction.git
cd House-Price-Prediction
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the ML Pipeline
```bash
python main.py
```

### Step 5: Launch Interactive Dashboard
```bash
streamlit run app.py
```

---

## 📸 Screenshots

> Add screenshots to the `images/` folder after running the project.

| Screenshot | Description |
|---|---|
| Dataset Preview | First 10 rows of the housing dataset |
| Correlation Heatmap | Feature correlation analysis |
| Price Distribution | Distribution of house prices |
| Model Comparison | R² score comparison of all 7 models |
| Actual vs Predicted | Scatter plot of predictions |
| Feature Importance | Top features affecting price |
| Dashboard | Interactive Streamlit app |

---

## 📁 Folder Structure

```
House-Price-Prediction/
├── data/                     # Raw and processed datasets
├── notebooks/                # Jupyter notebooks (optional)
├── src/                      # Source code modules
│   ├── __init__.py
│   ├── data_generator.py     # Synthetic data generation
│   ├── data_preprocessing.py # Data cleaning & encoding
│   ├── feature_engineering.py# Feature creation
│   ├── model_training.py     # Model training pipeline
│   ├── model_evaluation.py   # Evaluation metrics
│   ├── visualization.py      # All plotting functions
│   └── predictor.py          # Prediction interface
├── models/                   # Saved trained models
├── outputs/                  # Generated charts
├── images/                   # Screenshots for README
├── app.py                    # Streamlit dashboard
├── main.py                   # Main pipeline script
├── requirements.txt          # Dependencies
├── .gitignore                # Git exclusions
└── README.md                 # This file
```

---

## 🎓 Learning Outcomes

After completing this project, you will understand:

- ✅ End-to-end ML pipeline development
- ✅ Synthetic data generation with realistic distributions
- ✅ Data cleaning, preprocessing, and feature engineering
- ✅ Training and comparing multiple regression models
- ✅ Model evaluation using MAE, RMSE, R², and cross-validation
- ✅ Building interactive dashboards with Streamlit
- ✅ Data visualization with Matplotlib, Seaborn, and Plotly
- ✅ Git/GitHub workflow for portfolio projects

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## ⭐ Show Your Support

If you found this project helpful, please give it a ⭐ on GitHub!

---

## 👨‍💻 Author

**CH S K CHAITANYA**

[![GitHub](https://img.shields.io/badge/GitHub-CH--S--K--CHAITANYA-181717?style=for-the-badge&logo=github)](https://github.com/CH-S-K-CHAITANYA)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-chskchaitanya-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/chskchaitanya)

---

*Built by [CH S K CHAITANYA](https://github.com/CH-S-K-CHAITANYA) for Data Science & ML Portfolio*
