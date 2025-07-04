# Import all the necessary libraries
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import os
from sklearn.exceptions import NotFittedError

# Configure page
st.set_page_config(page_title="Water Quality Predictor", layout="wide")

# Custom CSS for better appearance
st.markdown("""
<style>
    .stNumberInput, .stTextInput {margin-bottom: 20px;}
    .stButton>button {background-color: #4CAF50; color: white;}
    .prediction-results {background-color: #f0f2f6; padding: 20px; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

# Error-handled model loading
@st.cache_resource
def load_model():
    try:
        if not os.path.exists("pollution_model.pkl"):
            raise FileNotFoundError("Model file 'pollution_model.pkl' not found")
        if not os.path.exists("model_columns.pkl"):
            raise FileNotFoundError("Columns file 'model_columns.pkl' not found")
            
        model = joblib.load("pollution_model.pkl")
        model_cols = joblib.load("model_columns.pkl")
        
        # Validate model structure
        if not hasattr(model, "predict"):
            raise ValueError("Loaded model doesn't have predict method")
        if not isinstance(model_cols, (list, np.ndarray)):
            raise TypeError("Model columns should be a list or array")
            
        return model, model_cols
        
    except Exception as e:
        st.error(f"🚨 Model loading failed: {str(e)}")
        st.stop()

# Load model with error handling
try:
    model, model_cols = load_model()
except Exception as e:
    st.error(f"Critical error: {str(e)}")
    st.stop()

# User interface
st.title("🌊 Water Pollutants Predictor")
st.markdown("Predict water quality parameters based on year and monitoring station")

# Input validation
def validate_inputs(year, station_id):
    errors = []
    if not station_id.strip():
        errors.append("Station ID cannot be empty")
    if year < 2000 or year > 2100:
        errors.append("Year must be between 2000-2100")
    return errors

# Prediction function with full error handling
def make_prediction(year, station_id):
    try:
        # Prepare input with validation
        input_df = pd.DataFrame({
            'year': [year],
            'id': [str(station_id).strip()]
        })
        
        # One-hot encode station ID
        input_encoded = pd.get_dummies(input_df, columns=['id'])
        
        # Ensure all expected columns exist
        missing_cols = set(model_cols) - set(input_encoded.columns)
        for col in missing_cols:
            input_encoded[col] = 0
            
        # Reorder columns to match model expectations
        input_encoded = input_encoded.reindex(columns=model_cols, fill_value=0)
        
        # Make prediction
        prediction = model.predict(input_encoded)
        
        if len(prediction[0]) != 6:
            raise ValueError(f"Expected 6 output values, got {len(prediction[0])}")
            
        return prediction[0]
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None

# User inputs with better formatting
col1, col2 = st.columns(2)
with col1:
    year_input = st.number_input(
        "Enter Year (2000-2100)",
        min_value=2000,
        max_value=2100,
        value=2023,
        step=1
    )
    
with col2:
    station_id = st.text_input(
        "Enter Station ID",
        value="1",
        placeholder="E.g. 1, 2, 3...",
        help="Enter the monitoring station identifier"
    )

# Prediction button and results
if st.button("Predict Water Quality", type="primary"):
    # Validate inputs
    validation_errors = validate_inputs(year_input, station_id)
    
    if validation_errors:
        for error in validation_errors:
            st.warning(error)
    else:
        with st.spinner("Analyzing water quality..."):
            predicted_values = make_prediction(year_input, station_id)
            
            if predicted_values is not None:
                pollutants = {
                    "O₂": predicted_values[0],
                    "NO₃": predicted_values[1],
                    "NO₂": predicted_values[2],
                    "SO₄": predicted_values[3],
                    "PO₄": predicted_values[4],
                    "Cl": predicted_values[5]
                }
                
                st.success("Prediction completed successfully!")
                
                # Display results in a nice layout
                st.subheader(f"🏭 Station {station_id} | 📅 Year {year_input}")
                
                cols = st.columns(2)
                for i, (pollutant, value) in enumerate(pollutants.items()):
                    with cols[i % 2]:
                        st.metric(
                            label=pollutant,
                            value=f"{value:.2f} mg/L",
                            help=f"Predicted {pollutant} concentration"
                        )
                
                # Visual indicator
                st.progress(100, text="Analysis complete")
