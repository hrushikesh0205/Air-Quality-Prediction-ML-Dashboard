import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from pathlib import Path
import joblib

# ==============================
# Page Configuration
# ==============================
st.set_page_config(page_title="Air Quality Prediction", layout="wide")

# ==============================
# Custom CSS Styling
# ==============================
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    h1, h2, h3 {
        color: #00BFFF;
    }
    .stButton button {
        background-color: #00BFFF;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        transition: none !important;
        box-shadow: none !important;
    }
    .stButton button:focus:not(:active) {
        color: white !important;
        background-color: #00BFFF !important;
        box-shadow: none !important;
    }
    .stDataFrame {
        border: 1px solid #333333;
        border-radius: 10px;
    }
    .nav-btn-selected {
        background-color: #1db0de33 !important;
        color: #00BFFF !important;
        font-weight: bold;
        border-radius: 8px;
        border: 1px solid #00BFFF !important;
        padding: 0.8em 2em;
        text-align: center;
    }
    .nav-btn {
        background-color: #F8F9FA !important;
        color: #1db0de !important;
        border-radius: 8px;
        border: 1px solid #E5E7EB !important;
        font-weight: 600;
        padding: 0.8em 2em;
        text-align: center;
        transition: none !important;
        box-shadow: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# Header Banner + Navigation
# ==============================
st.markdown("""
<div style='background-color:#001F3F;padding:15px;border-radius:10px;margin-bottom:10px'>
    <h1 style='color:white;text-align:center;'>🌍 Air Quality Prediction Dashboard</h1>
    <p style='color:#AAAAAA;text-align:center;margin-bottom:2px;'>Predict • Visualize • Detect Anomalies</p>
</div>
""", unsafe_allow_html=True)

nav_labels = ["PM2.5 Prediction", "Dataset Visualization", "Anomaly Detection"]

if "page" not in st.session_state:
    st.session_state.page = nav_labels[0]

nav_cols = st.columns(len(nav_labels))
for i, label in enumerate(nav_labels):
    if st.session_state.page == label:
        nav_cols[i].markdown(f"<div class='nav-btn-selected'>{label}</div>", unsafe_allow_html=True)
    else:
        if nav_cols[i].button(label, key=f"nav_btn_{i}", use_container_width=True):
            st.session_state.page = label

page = st.session_state.page

# ==============================
# Load Model Function
# ==============================
@st.cache_resource
def load_model():
    model_path = Path("model/air_quality_model.pkl")
    if model_path.exists():
        model = joblib.load(model_path)
        st.success("✅ Model loaded successfully!")
        return model
    else:
        st.warning("⚠️ Model file not found! Please run retrain_model.py first.")
        return None

model = load_model()

# ==============================
# Load & Merge Datasets
# ==============================
@st.cache_data
def load_all_data():
    folder = Path("dataset")
    files = [
        folder / "cleaned_Location_data_2021.xlsx",
        folder / "cleaned_Location_data_2022.xlsx",
        folder / "cleaned_Location_data_2023.xlsx"
    ]

    all_data = []
    for file in files:
        try:
            df = pd.read_excel(file, skiprows=1)
            df.columns = [
                "State / Union Territory",
                "City / town",
                "SO2 Annual Average",
                "NO2 Annual Average",
                "PM10 Annual Average",
                "PM2.5 Annual Average"
            ]
            df.replace(['NM', '-', ' '], np.nan, inplace=True)
            df.dropna(inplace=True)
            df.iloc[:, 2:] = df.iloc[:, 2:].astype(float)
            all_data.append(df)
        except Exception as e:
            st.warning(f"⚠️ Could not load {file.name}: {e}")

    merged = pd.concat(all_data, ignore_index=True)
    return merged

data = load_all_data()

# ==============================
# 1️⃣ PM2.5 Prediction Page
# ==============================
if page == "PM2.5 Prediction":
    st.header("📥 Enter Pollutant Annual Averages")

    # Separate features and target
    X = data[["SO2 Annual Average", "NO2 Annual Average", "PM10 Annual Average"]]
    y = data["PM2.5 Annual Average"]

    if model:
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
       # st.info(f"📊 **Model R² Score:** {r2:.3f} — indicates {(r2*100):.1f}% accuracy on the dataset.")

    # User input
    so2 = st.number_input("SO₂ Annual Average (µg/m³):", min_value=0.0, step=0.1)
    no2 = st.number_input("NO₂ Annual Average (µg/m³):", min_value=0.0, step=0.1)
    pm10 = st.number_input("PM₁₀ Annual Average (µg/m³):", min_value=0.0, step=0.1)

    if st.button("🔮 Predict PM2.5"):
        if model:
            input_data = np.array([[so2, no2, pm10]])
            prediction = model.predict(input_data)[0]

            if prediction <= 30:
                level, color = "Good", "🟢"
            elif prediction <= 60:
                level, color = "Moderate", "🟡"
            elif prediction <= 90:
                level, color = "Poor", "🟠"
            elif prediction <= 120:
                level, color = "Very Poor", "🔴"
            else:
                level, color = "Severe", "🟣"

            st.markdown(f"""
            <div style='padding:15px;border-radius:10px;background-color:#1E1E1E;text-align:center'>
                <h2>Predicted PM2.5: <span style='color:#00FFAA'>{round(prediction,2)} µg/m³</span></h2>
                <h3>Air Quality Level: <span style='color:#FFD700'>{color} {level}</span></h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("⚠️ Model not loaded. Please retrain it.")

# ==============================
# 2️⃣ Dataset Visualization Page
# ==============================
elif page == "Dataset Visualization":
    st.header("📊 Dataset Visualization")

    st.subheader("🧾 Dataset Preview")
    st.dataframe(data.head())

    st.subheader("📈 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(data.iloc[:, 2:].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("🎯 Interactive Visualization")
    feature = st.selectbox("Choose feature to compare with PM2.5:", data.columns[2:-1])
    fig = px.scatter(
        data,
        x=feature,
        y="PM2.5 Annual Average",
        color="State / Union Territory",
        title=f"{feature} vs PM2.5 Annual Average",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# ==============================
# 3️⃣ Anomaly Detection Page
# ==============================
elif page == "Anomaly Detection":
    st.header("⚠️ Anomaly Detection")

    method = st.radio(
        "Select Detection Method:",
        ["Z-Score Method", "Isolation Forest (ML-Based)"]
    )

    if method == "Z-Score Method":
        z_scores = np.abs(stats.zscore(data["PM2.5 Annual Average"]))
        threshold = 3
        anomalies = data[z_scores > threshold]

        st.subheader("Z-Score Based Anomalies")
        st.write(f"🔍 Total anomalies detected: {len(anomalies)}")
        st.dataframe(anomalies)

        fig = px.scatter(
            data,
            x="City / town",
            y="PM2.5 Annual Average",
            color=z_scores > threshold,
            title="Anomaly Detection using Z-Score",
            color_discrete_map={True: "red", False: "blue"},
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        model_iso = IsolationForest(contamination=0.05, random_state=42)
        pred = model_iso.fit_predict(data[["PM2.5 Annual Average"]])
        data["Anomaly"] = pred

        st.subheader("Isolation Forest Based Anomalies")
        st.write(f"🔍 Total anomalies detected: {len(data[data['Anomaly'] == -1])}")
        st.dataframe(data[data["Anomaly"] == -1])

        fig = px.scatter(
            data,
            x="City / town",
            y="PM2.5 Annual Average",
            color=data["Anomaly"].map({1: "Normal", -1: "Anomaly"}),
            title="Anomaly Detection using Isolation Forest",
            color_discrete_map={"Normal": "blue", "Anomaly": "red"},
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

# ==============================
# Footer
# ==============================
st.markdown(
    "<hr><p style='text-align: center; font-size: 13px; color: #808080;'>© 2025 | Developed by <b>HRUSHIKESH BHOIR</b></p>",
    unsafe_allow_html=True
)
