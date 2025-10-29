import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

st.title("🌦️ Weather Forecasting App")

# ============================
# Load dataset
# ============================
@st.cache_data
def load_data():
    df = pd.read_csv(r"D:\Weaather Forecast\Dataset\daily_training_table.csv")
    
    # Encode categorical (string) columns except 'date'
    for col in df.select_dtypes(include=["object"]).columns:
        if col != "date":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    
    return df

df = load_data()
st.subheader("📊 Sample Data")
st.write(df.head())

# ============================
# Features & Target
# ============================
drop_cols = ["date"]   # remove date from features
X = df.drop(columns=drop_cols + ["target_y"])
y = df["target_y"]

# ============================
# Train-test split
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================
# Train model
# ============================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "weather_model.pkl")

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

st.subheader("📉 Model Evaluation")
st.write(f"Mean Squared Error (MSE): {mse:.4f}")

# ============================
# Load model back
# ============================
loaded_model = joblib.load("weather_model.pkl")

st.subheader("🌤️ Make a Prediction")

# ============================
# Take user input dynamically
# ============================
input_data = {}
for col in X.columns:   # numeric + encoded categorical
    val = st.number_input(
        f"Enter value for {col}", 
        float(df[col].min()), 
        float(df[col].max()), 
        float(df[col].mean())
    )
    input_data[col] = val

input_df = pd.DataFrame([input_data])

# ============================
# Prediction + Human-readable label
# ============================
if st.button("Predict"):
    prediction = loaded_model.predict(input_df)[0]
    rounded_pred = int(round(prediction))   # convert to nearest category

    # ✅ Mapping (edit as per your dataset meaning)
    weather_labels = {
        0: "🌞 Sunny",
        1: "☁️ Cloudy",
        2: "🌧️ Rainy",
        3: "🌩️ Stormy"
    }

    weather_result = weather_labels.get(rounded_pred, f"Unknown ({rounded_pred})")

    st.success(f"🌡️ Predicted Weather for Tomorrow: **{weather_result}**")
