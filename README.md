# Hotel Booking Cancellation Prediction

A Machine Learning project to predict hotel booking cancellations using the Hotel Booking Demand Dataset.

## Project Overview

This project applies multiple machine learning algorithms to predict whether a hotel booking will be canceled, and includes an optional regression task to predict the Average Daily Rate (ADR).

## Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- Naive Bayes
- K-Nearest Neighbors
- Support Vector Machine
- Neural Network (MLP)


## Project Structure

```
├── hotel_bookings.csv                 # Dataset
├── hotel_booking.ipynb                # Main Jupyter Notebook
├── app.py                             # Streamlit GUI
├── best_cancellation_model.pkl        # Saved best model
├── scaler.pkl                         # Saved scaler
├── model_columns.pkl                  # Saved feature columns
└── requirements.txt                   # Python dependencies
```

## Live Demo

[Click here to try the app](https://hotel-booking-cancellation-mo-khalid-1.streamlit.app/)

## How to Run Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

## Technologies

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit
