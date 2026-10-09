import streamlit as st
import pandas as pd
import joblib

# 1. Saved model ko load karein
model = joblib.load('stock_model.pkl')

st.title("Stock Market Predictor")
st.write("Enter today's market data to predict tomorrow's price change.")

# 2. User se input lene ke liye UI banayein
open_price = st.number_input("Open Price", value=100.0)
high_price = st.number_input("High Price", value=105.0)
low_price = st.number_input("Low Price", value=95.0)
close_price = st.number_input("Close Price", value=102.0)
volume = st.number_input("Volume", value=10000)

# 3. Prediction button
if st.button("Predict Next Day"):
    # Input data ko wahi dataframe format dein jispar model train hua tha
    input_data = pd.DataFrame([[open_price, high_price, low_price, close_price, volume]],
                              columns=['Open', 'High', 'Low', 'Close', 'Volume'])

    # Model se predict karein
    prediction = model.predict(input_data)
    st.success(f"Predicted Price Change: {prediction[0]:.2f}")