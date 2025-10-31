# evaluate_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# ==============================
# Load merged dataset
# ==============================
print("📂 Loading merged dataset...")
data = pd.read_csv("dataset/merged_air_quality.csv")

# Drop missing values just in case
data.dropna(inplace=True)

# ==============================
# Prepare data
# ==============================
X = data[["SO2 Annual Average", "NO2 Annual Average", "PM10 Annual Average"]]
y = data["PM2.5 Annual Average"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# Train model
# ==============================
model = LinearRegression()
model.fit(X_train, y_train)

# ==============================
# Predictions
# ==============================
y_pred = model.predict(X_test)

# ==============================
# Evaluate performance
# ==============================
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n📊 Model Performance on Test Set")
print(f"R² Score (Accuracy): {r2:.3f}")
print(f"Mean Absolute Error: {mae:.3f}")
print(f"Root Mean Squared Error: {rmse:.3f}")

# ==============================
# Visualization
# ==============================
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color="blue", label="Predictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color="red", linestyle="--", label="Perfect Fit")
plt.title("Predicted vs Actual PM2.5 Values")
plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.legend()
plt.grid(True)
plt.show()
