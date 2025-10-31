# retrain_model.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
from pathlib import Path

# ==============================
# Load merged dataset
# ==============================
print("📂 Loading merged dataset...")
data = pd.read_csv("dataset/merged_air_quality.csv")
data.dropna(inplace=True)

# ==============================
# Prepare data
# ==============================
X = data[["SO2 Annual Average", "NO2 Annual Average", "PM10 Annual Average"]]
y = data["PM2.5 Annual Average"]

# ==============================
# Train model
# ==============================
model = LinearRegression()
model.fit(X, y)

# ==============================
# Save model
# ==============================
model_dir = Path("model")
model_dir.mkdir(exist_ok=True)
model_path = model_dir / "air_quality_model.pkl"

joblib.dump(model, model_path)

print("\n✅ Model retrained successfully!")
print(f"💾 Saved model to: {model_path}")
print(f"📊 Trained on {len(data)} records.")
