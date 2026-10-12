"""
Air Quality Prediction System - Streamlit Interactive Web Application
Built with Scikit-Learn Pipeline and Joblib.
Features:
- Real-time Air Quality Status Prediction (Good/Moderate vs Poor/Hazardous)
- Model Probability Gauges & Severity Ratings
- Actionable Health & Environmental Guidance
- Interactive Exploratory Data Analysis & Model Comparison Dashboard
"""

import json
import os

import joblib
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="AeroGuard | Air Quality Intelligence",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Premium Aesthetics
st.markdown(
    """<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    padding: 24px 30px;
    border-radius: 16px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(30, 60, 114, 0.15);
}

.main-header h1 {
    color: white !important;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}

.main-header p {
    color: #dbe4ff;
    font-size: 1.05rem;
    margin-top: 6px;
    margin-bottom: 0;
}

.card-metric {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    margin-bottom: 16px;
}

.status-badge-good {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white;
    padding: 16px 24px;
    border-radius: 12px;
    font-weight: 700;
    font-size: 1.3rem;
    text-align: center;
    box-shadow: 0 6px 18px rgba(16, 185, 129, 0.25);
}

.status-badge-poor {
    background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
    color: white;
    padding: 16px 24px;
    border-radius: 12px;
    font-weight: 700;
    font-size: 1.3rem;
    text-align: center;
    box-shadow: 0 6px 18px rgba(239, 68, 68, 0.25);
}

.stButton>button {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    color: white;
    font-weight: 600;
    padding: 12px 28px;
    border-radius: 10px;
    border: none;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3);
    transition: all 0.2s ease;
    width: 100%;
    font-size: 1.05rem;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4);
}
</style>""",
    unsafe_allow_html=True,
)


# Load trained pipeline
@st.cache_resource
def load_pipeline():
    model_paths = ["best_air_quality_model.joblib", "model.joblib"]
    for path in model_paths:
        if os.path.exists(path):
            return joblib.load(path)
    return None


# Load evaluation metrics if available
@st.cache_data
def load_metrics():
    if os.path.exists("model_evaluation_results.json"):
        with open("model_evaluation_results.json", "r") as f:
            return json.load(f)
    return None


# Load dataset for exploration
@st.cache_data
def load_data():
    if os.path.exists("air_quality_data.csv"):
        return pd.read_csv("air_quality_data.csv")
    return None


model = load_pipeline()
metrics_data = load_metrics()
df = load_data()

# App Header
st.markdown(
    """<div class="main-header">
    <h1>🍃 AeroGuard: Intelligent Air Quality Predictor</h1>
    <p>Predict real-time atmospheric hazards using machine learning pipelines
    trained on environmental & meteorological sensors.</p>
</div>""",
    unsafe_allow_html=True,
)

# Navigation tabs
tab_pred, tab_dash, tab_models, tab_dataset = st.tabs(
    [
        "🎯 Real-Time Predictor",
        "📊 EDA & Visualizations",
        "🧪 Model Comparison & Performance",
        "📁 Dataset Explorer",
    ]
)

# ==========================================
# TAB 1: REAL-TIME PREDICTOR
# ==========================================
with tab_pred:
    st.subheader("1. Atmospheric & Meteorological Input Parameters")
    st.write(
        "Adjust the environmental sensors below to forecast air quality status."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 🌫️ Particulate Matter Concentrations")
        pm25 = st.slider(
            "PM2.5 (Fine Particles, µg/m³)",
            min_value=5.0,
            max_value=350.0,
            value=35.0,
            step=1.0,
            help="WHO guideline: <15 µg/m³; Moderate: 15-60; Severe: >60",
        )
        pm10 = st.slider(
            "PM10 (Coarse Dust Particles, µg/m³)",
            min_value=10.0,
            max_value=500.0,
            value=75.0,
            step=1.0,
            help="WHO guideline: <45 µg/m³; Moderate: 45-100; Severe: >100",
        )

        st.markdown("##### 🏭 Chemical & Gaseous Pollutants")
        no2 = st.slider(
            "NO2 (Nitrogen Dioxide, µg/m³)",
            min_value=2.0,
            max_value=180.0,
            value=38.0,
            step=1.0,
            help="Primarily from vehicle emissions and power stations.",
        )
        so2 = st.slider(
            "SO2 (Sulfur Dioxide, µg/m³)",
            min_value=1.0,
            max_value=150.0,
            value=25.0,
            step=1.0,
            help="Associated with coal and industrial fuel combustion.",
        )

    with col2:
        st.markdown("##### 🚗 Combustion & Photochemical Pollutants")
        co = st.slider(
            "CO (Carbon Monoxide, mg/m³)",
            min_value=0.1,
            max_value=12.0,
            value=1.5,
            step=0.1,
            help="Toxic gas from incomplete combustion of fuels.",
        )
        o3 = st.slider(
            "O3 (Ground-level Ozone, µg/m³)",
            min_value=5.0,
            max_value=220.0,
            value=50.0,
            step=1.0,
            help="Secondary pollutant formed via sunlight and heat reactions.",
        )

        st.markdown("##### 🌦️ Weather & Atmospheric Context")
        sub_c1, sub_c2 = st.columns(2)
        with sub_c1:
            temperature = st.slider(
                "Temperature (°C)",
                min_value=5.0,
                max_value=48.0,
                value=26.0,
                step=0.5,
            )
            humidity = st.slider(
                "Humidity (%)",
                min_value=15.0,
                max_value=98.0,
                value=60.0,
                step=1.0,
            )
        with sub_c2:
            area_type = st.selectbox(
                "Area Zone",
                ["Residential", "Commercial", "Industrial", "Rural"],
                index=0,
            )
            season = st.selectbox(
                "Season",
                ["Winter", "Summer", "Monsoon", "Autumn"],
                index=3,
            )

    st.markdown("---")
    predict_btn = st.button(
        "🔍 Predict Air Quality Status", use_container_width=True
    )

    if predict_btn:
        if model is None:
            st.error(
                "⚠️ Trained model not found! Please run `train_models.py`."
            )
        else:
            input_dict = {
                "PM2.5": [pm25],
                "PM10": [pm10],
                "NO2": [no2],
                "SO2": [so2],
                "CO": [co],
                "O3": [o3],
                "Temperature": [temperature],
                "Humidity": [humidity],
                "Area_Type": [area_type],
                "Season": [season],
            }
            input_df = pd.DataFrame(input_dict)

            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0]
            prob_poor = prob[1]
            prob_good = prob[0]

            st.markdown("### 📋 Prediction Diagnostic & Assessment")
            res_col1, res_col2 = st.columns([1.2, 1])

            with res_col1:
                if pred == 1:
                    st.markdown(
                        """<div class="status-badge-poor">
                        🚨 HAZARDOUS / POOR AIR QUALITY (Class 1)
                        <div style="font-size: 0.95rem; font-weight: 500; margin-top: 6px;">
                            Smog Alert: Elevated pollutant concentrations pose health hazards.
                        </div>
                    </div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """<div class="status-badge-good">
                        ✅ SAFE / GOOD TO MODERATE AIR QUALITY (Class 0)
                        <div style="font-size: 0.95rem; font-weight: 500; margin-top: 6px;">
                            Air quality is within acceptable thresholds for public activities.
                        </div>
                    </div>""",
                        unsafe_allow_html=True,
                    )

                st.write("")
                st.markdown("##### Classification Probability Confidence")
                st.progress(float(prob_poor))
                m1, m2 = st.columns(2)
                m1.metric("P(Poor/Hazardous)", f"{prob_poor*100:.1f}%")
                m2.metric("P(Good/Moderate)", f"{prob_good*100:.1f}%")

            with res_col2:
                st.markdown("##### 🛡️ Recommended Health Precautions")
                if pred == 1:
                    st.warning(
                        "- **Vulnerable Groups**: Stay indoors.\n"
                        "- **Protection**: Wear an **N95 respirator** outside.\n"
                        "- **Indoor Air**: Seal windows; run air purifiers.\n"
                        "- **Exercise**: Avoid strenuous outdoor jogging."
                    )
                else:
                    st.success(
                        "- **General Public**: Ideal conditions for recreation.\n"
                        "- **Ventilation**: Natural ventilation is recommended.\n"
                        "- **Vulnerable Groups**: No restrictions necessary."
                    )

            st.markdown("##### 🔬 Sensor Values vs WHO Clean Air Standards")
            pollutants_summary = pd.DataFrame(
                {
                    "Parameter": [
                        "PM2.5 (µg/m³)",
                        "PM10 (µg/m³)",
                        "NO2 (µg/m³)",
                        "SO2 (µg/m³)",
                        "CO (mg/m³)",
                        "O3 (µg/m³)",
                    ],
                    "Input Value": [pm25, pm10, no2, so2, co, o3],
                    "Standard Safe Limit": [
                        "15.0",
                        "45.0",
                        "25.0",
                        "40.0",
                        "4.0",
                        "100.0",
                    ],
                    "Status": [
                        "⚠️ Exceeds Limit" if pm25 > 15.0 else "✅ Normal",
                        "⚠️ Exceeds Limit" if pm10 > 45.0 else "✅ Normal",
                        "⚠️ Exceeds Limit" if no2 > 25.0 else "✅ Normal",
                        "⚠️ Exceeds Limit" if so2 > 40.0 else "✅ Normal",
                        "⚠️ Exceeds Limit" if co > 4.0 else "✅ Normal",
                        "⚠️ Exceeds Limit" if o3 > 100.0 else "✅ Normal",
                    ],
                }
            )
            st.table(pollutants_summary)

# ==========================================
# TAB 2: EDA & VISUALIZATIONS
# ==========================================
with tab_dash:
    st.subheader("Exploratory Data Analysis (EDA) Charts")
    st.write("Visualizations generated from atmospheric sensor data.")

    vcol1, vcol2 = st.columns(2)
    with vcol1:
        if os.path.exists("plots/class_distribution.png"):
            st.image(
                "plots/class_distribution.png",
                caption="Fig 1: Target Class Balance (Safe vs Hazardous Air)",
            )
        else:
            st.info("Run `train_models.py` to generate plots.")

        if os.path.exists("plots/confusion_matrices.png"):
            st.image(
                "plots/confusion_matrices.png",
                caption="Fig 3: Confusion Matrices Across ML Models",
            )

    with vcol2:
        if os.path.exists("plots/correlation_heatmap.png"):
            st.image(
                "plots/correlation_heatmap.png",
                caption="Fig 2: Correlation Heatmap Between Pollutants",
            )

        if os.path.exists("plots/roc_curves.png"):
            st.image(
                "plots/roc_curves.png",
                caption="Fig 4: Receiver Operating Characteristic (ROC) Curves",
            )

# ==========================================
# TAB 3: MODEL COMPARISON & METRICS
# ==========================================
with tab_models:
    st.subheader("Model Performance & Cross-Validation Benchmarks")
    if metrics_data:
        st.write(
            f"**Winning Model (5-Fold CV):** `{metrics_data['best_model']}`"
        )

        test_m = metrics_data["test_metrics"]
        cv_m = metrics_data["cv_results"]

        table_rows = []
        for name in test_m:
            table_rows.append(
                {
                    "Algorithm": name,
                    "5-Fold CV F1": (
                        f"{cv_m[name]['cv_f1_mean']:.4f} ± "
                        f"{cv_m[name]['cv_f1_std']:.4f}"
                    ),
                    "5-Fold CV ROC-AUC": (
                        f"{cv_m[name]['cv_auc_mean']:.4f} ± "
                        f"{cv_m[name]['cv_auc_std']:.4f}"
                    ),
                    "Test Accuracy": f"{test_m[name]['Accuracy']*100:.2f}%",
                    "Test Precision": f"{test_m[name]['Precision']:.4f}",
                    "Test Recall": f"{test_m[name]['Recall']:.4f}",
                    "Test F1-Score": f"{test_m[name]['F1_Score']:.4f}",
                    "Test ROC-AUC": f"{test_m[name]['ROC_AUC']:.4f}",
                    "False Negatives (FN)": test_m[name]["FN"],
                    "False Positives (FP)": test_m[name]["FP"],
                }
            )
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        if os.path.exists("plots/model_comparison.png"):
            st.image(
                "plots/model_comparison.png",
                caption="Fig 5: Multi-Metric Model Benchmark Comparison",
            )

        st.markdown(
            "> **Critical Error Analysis:** In public health air pollution "
            "monitoring, a **False Negative (FN)** (predicting clean air when "
            "air is actually hazardous) fails to trigger smog advisories. A "
            "**False Positive (FP)** merely triggers precautionary masks. "
            "Hence, **Recall** and **F1-Score** matter most."
        )
    else:
        st.info("Metrics not loaded. Please execute `train_models.py`.")

# ==========================================
# TAB 4: DATASET EXPLORER
# ==========================================
with tab_dataset:
    st.subheader("Air Quality Sensor Dataset Explorer")
    if df is not None:
        st.write(
            f"Total Rows: **{len(df)}** | Total Features: **{df.shape[1]}**"
        )
        st.dataframe(df.head(50), use_container_width=True)
        st.markdown("##### Descriptive Statistics")
        st.dataframe(df.describe(), use_container_width=True)
    else:
        st.warning("Dataset not found. Run `generate_dataset.py` first.")
