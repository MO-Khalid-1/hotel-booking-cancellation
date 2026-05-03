"""
Hotel Booking Cancellation Prediction - Streamlit GUI
Run this application with: streamlit run app.py

Requirements:
- best_cancellation_model.pkl
- scaler.pkl
- model_columns.pkl

All three files must be in the same folder as this script.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------------------------------
# Page configuration
# -----------------------------------------------------------
st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    layout="centered"
)

st.title("Hotel Booking Cancellation Predictor")
st.markdown(
    "Enter the booking details below and click **Predict** to see whether "
    "the booking is likely to be canceled."
)
st.markdown("---")

# -----------------------------------------------------------
# Load saved model artifacts
# -----------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_cancellation_model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("model_columns.pkl")
    return model, scaler, columns

try:
    model, scaler, model_columns = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"Model file not found: {e}\n\n"
        "Make sure best_cancellation_model.pkl, scaler.pkl, and "
        "model_columns.pkl are in the same folder as app.py."
    )
    st.stop()

# -----------------------------------------------------------
# Input form
# -----------------------------------------------------------
st.subheader("Booking Information")

col1, col2 = st.columns(2)

with col1:
    hotel = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
    lead_time = st.number_input("Lead Time (days)", min_value=0, max_value=737, value=60)
    adults = st.number_input("Number of Adults", min_value=0, max_value=10, value=2)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    babies = st.number_input("Number of Babies", min_value=0, max_value=10, value=0)

with col2:
    adr = st.number_input("Average Daily Rate (ADR)", min_value=0.0, max_value=5000.0, value=100.0)
    special_requests = st.number_input("Number of Special Requests", min_value=0, max_value=5, value=0)
    deposit_type = st.selectbox("Deposit Type", ["No Deposit", "Non Refund", "Refundable"])
    customer_type = st.selectbox("Customer Type", ["Transient", "Contract", "Transient-Party", "Group"])
    total_nights = st.number_input("Total Nights", min_value=0, max_value=60, value=3)

# Additional fields (less prominent)
st.markdown("**Additional Details**")
col3, col4 = st.columns(2)

with col3:
    arrival_month = st.selectbox("Arrival Month", [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ])
    market_segment = st.selectbox("Market Segment", [
        "Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups",
        "Complementary", "Aviation"
    ])
    meal = st.selectbox("Meal Type", ["BB", "HB", "FB", "SC", "Undefined"])

with col4:
    previous_cancellations = st.number_input("Previous Cancellations", min_value=0, max_value=26, value=0)
    required_parking = st.number_input("Required Parking Spaces", min_value=0, max_value=8, value=0)
    booking_changes = st.number_input("Booking Changes", min_value=0, max_value=21, value=0)

# -----------------------------------------------------------
# Feature construction (matches training pipeline)
# -----------------------------------------------------------
def build_input_row():
    # Encode categorical values as label-encoded integers
    hotel_map = {"City Hotel": 0, "Resort Hotel": 1}
    deposit_map = {"No Deposit": 1, "Non Refund": 2, "Refundable": 0}
    customer_map = {"Contract": 0, "Group": 1, "Transient": 2, "Transient-Party": 3}
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5,
        "June": 6, "July": 7, "August": 8, "September": 9,
        "October": 10, "November": 11, "December": 12
    }
    segment_map = {
        "Aviation": 0, "Complementary": 1, "Corporate": 2, "Direct": 3,
        "Groups": 4, "Offline TA/TO": 5, "Online TA": 6
    }
    meal_map = {"BB": 0, "FB": 1, "HB": 2, "SC": 3, "Undefined": 4}
    season_map = {
        "December": 0, "January": 0, "February": 0,
        "March": 1, "April": 1, "May": 1,
        "June": 2, "July": 2, "August": 2,
        "September": 3, "October": 3, "November": 3
    }

    # Derived features (matches notebook feature engineering)
    total_guests = adults + children + babies
    has_prev_cancel = 1 if previous_cancellations > 0 else 0
    is_family = 1 if (children > 0 or babies > 0) else 0
    booking_season = season_map[arrival_month]
    has_special = 1 if special_requests > 0 else 0

    # Build a dictionary matching training columns as closely as possible
    row = {
        "hotel": hotel_map[hotel],
        "lead_time": lead_time,
        "arrival_date_year": 2024,
        "arrival_date_month": month_map[arrival_month],
        "arrival_date_week_number": 27,
        "arrival_date_day_of_month": 1,
        "stays_in_weekend_nights": max(0, total_nights - int(total_nights * 5 / 7)),
        "stays_in_week_nights": int(total_nights * 5 / 7),
        "adults": adults,
        "children": children,
        "babies": babies,
        "meal": meal_map[meal],
        "country": 135,           # PRT as default (most frequent in training data)
        "market_segment": segment_map[market_segment],
        "distribution_channel": 2,
        "is_repeated_guest": 0,
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": 0,
        "assigned_room_type": 0,
        "booking_changes": booking_changes,
        "deposit_type": deposit_map[deposit_type],
        "agent": 0,
        "company": 0,
        "days_in_waiting_list": 0,
        "customer_type": customer_map[customer_type],
        "required_car_parking_spaces": required_parking,
        "total_of_special_requests": special_requests,
        "total_nights": total_nights,
        "total_guests": total_guests,
        "has_previous_cancellation": has_prev_cancel,
        "is_family_booking": is_family,
        "booking_season": booking_season,
        "has_special_request": has_special
    }
    return row

# -----------------------------------------------------------
# Determine if model needs scaling
# -----------------------------------------------------------
SCALE_SENSITIVE_MODELS = ["LogisticRegression", "KNeighborsClassifier",
                           "SVC", "MLPClassifier"]

def needs_scaling(model_obj):
    return type(model_obj).__name__ in SCALE_SENSITIVE_MODELS

# -----------------------------------------------------------
# Predict button
# -----------------------------------------------------------
st.markdown("---")
if st.button("Predict"):
    row = build_input_row()
    input_df = pd.DataFrame([row])

    # Align columns with training data
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_columns]

    # Scale if required
    if needs_scaling(model):
        input_array = scaler.transform(input_df)
    else:
        input_array = input_df.values

    prediction = model.predict(input_array)[0]

    # Display result
    st.markdown("### Prediction Result")
    if prediction == 1:
        st.error("Booking is likely to be **CANCELED**")
    else:
        st.success("Booking is likely **NOT to be canceled**")

    # Show probability if supported
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_array)[0]
        cancel_prob = prob[1] * 100
        not_cancel_prob = prob[0] * 100
        st.markdown(f"**Cancellation Probability:** {cancel_prob:.1f}%")
        st.markdown(f"**Not Canceled Probability:** {not_cancel_prob:.1f}%")
        st.progress(int(cancel_prob))

# -----------------------------------------------------------
# Footer
# -----------------------------------------------------------
st.markdown("---")
st.caption(
    "Hotel Booking Cancellation Predictor - Machine Learning for Data Analytics Final Project. "
    "Predictions are based on a trained machine learning model and are not guaranteed to be accurate."
)
