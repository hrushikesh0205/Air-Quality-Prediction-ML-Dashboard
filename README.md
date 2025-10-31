# 🌍 Air Quality Prediction Dashboard

**Developed by Hrushikesh Bhoir | Department of Computer Engineering**  

---

## 📄 Project Summary

The **Air Quality Prediction Dashboard** is an interactive web application built using **Machine Learning (Multiple Linear Regression)** and **Streamlit**.  
It predicts **PM2.5 Annual Average** concentration based on major pollutants — **SO₂**, **NO₂**, and **PM₁₀**, helping to analyze and understand air quality patterns across cities.

The system also integrates **Anomaly Detection** (using both **Z-Score** and **Isolation Forest**) to identify unusually high or low pollution levels.  
Users can explore pollutant relationships, visualize data through heatmaps and scatter plots, and predict future air quality trends in real time.

---

## 🚀 Key Features

### 🔹 PM2.5 Prediction
- Predicts PM2.5 concentration using **SO₂**, **NO₂**, and **PM₁₀** values.  
- Categorizes air quality as **Good**, **Moderate**, **Poor**, **Very Poor**, or **Severe**.  

### 🔹 Dataset Visualization
- Displays a clean, tabular preview of air quality datasets (2021–2023).  
- Provides **correlation heatmaps** to show relationships among pollutants.  
- Includes **interactive scatter plots** for visual comparison between pollutants and PM2.5.

### 🔹 Anomaly Detection
- **Z-Score Method:** Detects cities/towns with abnormally high or low PM2.5 values.  
- **Isolation Forest (ML-Based):** Identifies complex pollution outliers using machine learning.

### 🔹 Model Training & Evaluation
- Dataset merging, cleaning, and retraining scripts for maintaining data accuracy.  
- Achieved an **R² Score ≈ 0.80** and **MAE ≈ 0.91**, showing strong predictive performance.  

---

## 🧠 Tech Stack

| Component | Technology Used |
|------------|-----------------|
| **Language** | Python |
| **Frontend/UI** | Streamlit |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Machine Learning** | Scikit-learn (Linear Regression, Isolation Forest) |
| **Anomaly Detection** | Z-Score, Isolation Forest |
| **Storage** | Local Datasets (Excel/CSV) |
| **IDE** | Visual Studio Code |

---

## 📂 Project Structure

Air_Quality_Prediction/
│
├── app.py # Main Streamlit dashboard
├── merge_and_clean.py # Script to merge and clean datasets
├── retrain_model.py # Retrains model using latest merged data
├── evaluate_model.py # Evaluates and plots model accuracy
│
├── model/
│ └── air_quality_model.pkl # Saved ML model file
│
├── dataset/
│ ├── cleaned_Location_data_2021.xlsx
│ ├── cleaned_Location_data_2022.xlsx
│ ├── cleaned_Location_data_2023.xlsx
│ └── merged_air_quality.csv
│
├── requirements.txt # Python dependencies
└── README.md # Project documentation