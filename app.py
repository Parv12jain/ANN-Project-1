import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# ----------------------------------------
# PAGE CONFIG & CUSTOM STYLE
# ----------------------------------------
st.set_page_config(page_title="ANN Prediction App", layout="centered")

st.markdown("""
<style>
body {
    background: #0d1117;
    color: white;
}
.header {
    background: linear-gradient(90deg, #4b79ff, #6cd0ff);
    padding: 22px;
    border-radius: 12px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0 5px 18px rgba(0,0,0,0.15);
}
.card {
    background: #161b22;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.10);
    margin-top: 20px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>🔮 ANN Model Prediction App</h1><p>Powered by annmodel.h5</p></div>', unsafe_allow_html=True)

# ----------------------------------------
# LOAD THE KERAS MODEL
# ----------------------------------------
try:
    model = load_model("annmodel.h5")
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ----------------------------------------
# DEFINE YOUR FEATURES HERE
# ----------------------------------------
FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Geography_Germany",
    "Geography_Spain",
    "Gender_Male"
]

st.subheader("Enter the input values")

# ----------------------------------------
# INPUT FIELDS (Sidebar)
# ----------------------------------------
st.sidebar.title("Input Parameters")

inputs = {}

for feat in FEATURES:
    inputs[feat] = st.sidebar.number_input(f"{feat}", value=0.0, format="%.4f")

# ----------------------------------------
# PREDICTION BUTTON
# ----------------------------------------
pred_value = None

if st.button("🔍 Predict"):
    try:
        x = np.array([list(inputs.values())])
        pred = model.predict(x)
        pred_value = float(pred[0][0]) if pred.shape == (1, 1) else pred[0]

        # FINAL OUTPUT CARD (Gradient Only)
        st.markdown(f"""
            <div class="card" style="background: linear-gradient(135deg, #4b79ff, #6cd0ff); color:white;">
                <h2>Prediction Result:</h2>
                <h2 style="font-size:28px; font-weight:700;">🧠 Output: {pred_value}</h2>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ----------------------------------------
# FOOTER
# ----------------------------------------
st.markdown("""
<div style="text-align:center; margin-top:40px; color:#8b949e;">
Made with ❤️ using Streamlit & Keras ANN Model.
</div>
""", unsafe_allow_html=True)
