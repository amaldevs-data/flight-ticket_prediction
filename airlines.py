import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load saved model objects
model = joblib.load("project.pkl")
scaler = joblib.load("scaler.pkl")

le  = joblib.load("le.pkl")     # airline
le1 = joblib.load("le1.pkl")    # flight
le2 = joblib.load("le2.pkl")    # source_city
le3 = joblib.load("le3.pkl")    # departure_time
le4 = joblib.load("le4.pkl")    # stops
le5 = joblib.load("le5.pkl")    # arrival_time
le6 = joblib.load("le6.pkl")    # destination_city
le7 = joblib.load("le7.pkl")    # class

st.title("Flight Price Prediction App (Regression)")
st.write("Predict the price of a flight ticket using trained ML model.")

# Input UI
airline = st.selectbox("Airline", le.classes_)
flight = st.selectbox("Flight Code", le1.classes_)
source_city = st.selectbox("Source City", le2.classes_)
departure_time = st.selectbox("Departure Time", le3.classes_)
stops = st.selectbox("Stops", le4.classes_)
arrival_time = st.selectbox("Arrival Time", le5.classes_)
destination_city = st.selectbox("Destination City", le6.classes_)
cls = st.selectbox("Class", le7.classes_)

duration = st.number_input("Duration (in hours)", min_value=0.0)
days_left = st.number_input("Days Left for Journey", min_value=0)

# Encode inputs
airline_enc = le.transform([airline])[0]
flight_enc = le1.transform([flight])[0]
source_city_enc = le2.transform([source_city])[0]
departure_time_enc = le3.transform([departure_time])[0]
stops_enc = le4.transform([stops])[0]
arrival_time_enc = le5.transform([arrival_time])[0]
destination_city_enc = le6.transform([destination_city])[0]
class_enc = le7.transform([cls])[0]

# Create input row in correct order
input_data = np.array([[airline_enc, flight_enc, source_city_enc,
                        departure_time_enc, stops_enc, arrival_time_enc,
                        destination_city_enc, class_enc, duration, days_left]])

# Scale numeric features
input_scaled = scaler.transform(input_data)

if st.button("Predict Price"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"Predicted Flight Price: ₹ {prediction:,.2f}")
