# tradein_model/server.py - Standalone ML server for trade-in valuation
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import json
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, 'models')

print("=" * 60)
print("Trade-In Valuation ML Server")
print("=" * 60)
print(f"Script directory: {SCRIPT_DIR}")
print(f"Model directory: {MODEL_DIR}")
print(f"Model exists: {os.path.exists(MODEL_DIR)}")

# Global variables for models
model = None
scaler = None
features = None
numerical_cols = None
label_encoders = {}
categorical_cols = ['make', 'model', 'condition', 'body_type', 'fuel_type', 'transmission']

def load_models():
    """Load all trade-in models and encoders"""
    global model, scaler, features, numerical_cols, label_encoders
    
    try:
        print("\nLoading trade-in models...")
        
        # Load model
        model_path = os.path.join(MODEL_DIR, 'tradein_model.pkl')
        if not os.path.exists(model_path):
            print(f"⚠️ Model file not found: {model_path}")
            return False
        
        model = joblib.load(model_path)
        print("✓ Model loaded successfully")
        
        # Load scaler
        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print("✓ Scaler loaded")
        
        # Load features
        features_path = os.path.join(MODEL_DIR, 'features.pkl')
        if os.path.exists(features_path):
            features = joblib.load(features_path)
            print("✓ Features loaded")
        
        # Load numerical columns
        num_cols_path = os.path.join(MODEL_DIR, 'numerical_cols.pkl')
        if os.path.exists(num_cols_path):
            numerical_cols = joblib.load(num_cols_path)
            print("✓ Numerical columns loaded")
        
        # Load label encoders
        for col in categorical_cols:
            encoder_path = os.path.join(MODEL_DIR, f'encoder_{col}.pkl')
            if os.path.exists(encoder_path):
                label_encoders[col] = joblib.load(encoder_path)
                print(f"✓ Encoder for {col} loaded")
        
        print("\n✅ All trade-in models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        import traceback
        traceback.print_exc()
        return False

def calculate_fallback_value(make, model_name, year, mileage, condition):
    """Fallback calculation if ML model fails"""
    # Base values by brand (NZD)
    brand_values = {
        'Toyota': 28000, 'Honda': 26000, 'Mazda': 27000, 'Subaru': 29000,
        'Nissan': 25000, 'Ford': 30000, 'BMW': 55000, 'Mercedes': 58000,
        'Audi': 52000, 'Hyundai': 24000, 'Kia': 23500, 'Volkswagen': 31000,
        'Tesla': 65000, 'Porsche': 120000, 'Ferrari': 400000, 'Lamborghini': 350000,
        'Chevrolet': 32000, 'Dodge': 35000, 'Jeep': 33000, 'Volvo': 38000,
        'Lexus': 45000, 'Jaguar': 48000, 'Land Rover': 52000, 'Mitsubishi': 22000
    }
    
    base_value = brand_values.get(make, 25000)
    
    # Age depreciation (10% per year)
    age = 2025 - year
    age_factor = max(0.3, 1.0 - (age * 0.10))
    
    # Mileage adjustment
    mileage_factor = max(0.6, 1.0 - (mileage / 200000.0))
    
    # Condition factor
    condition_factors = {
        'Excellent': 1.0, 'Very Good': 0.9, 'Good': 0.8,
        'Fair': 0.65, 'Poor': 0.5
    }
    condition_factor = condition_factors.get(condition, 0.7)
    
    estimated_value = base_value * age_factor * mileage_factor * condition_factor
    estimated_value = max(1000, min(150000, estimated_value))
    estimated_value = round(estimated_value / 100) * 100
    
    return estimated_value

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint that receives car details and returns estimated trade-in value
    """
    try:
        data = request.get_json()
        logger.info(f"Received prediction request: {data}")
        
        make = data.get('make')
        model_name = data.get('model')
        year = int(data.get('year'))
        mileage = int(data.get('mileage'))
        condition = data.get('condition')
        body_type = data.get('body_type', 'Sedan')
        fuel_type = data.get('fuel_type', 'Petrol')
        transmission = data.get('transmission', 'Automatic')
        owners = int(data.get('owners', 1))
        engine_size = float(data.get('engine_size', 2.0))
        
        # Check if ML model is available
        if model is None:
            logger.warning("ML model not available, using fallback calculation")
            estimated_value = calculate_fallback_value(make, model_name, year, mileage, condition)
            return jsonify({
                'success': True,
                'predicted_value': float(estimated_value),
                'model': 'fallback_v1',
                'message': 'Using fallback calculation (ML model not loaded)'
            })
        
        # Calculate age
        current_year = 2025
        age = current_year - year
        
        # Condition scores
        condition_scores = {'Excellent': 5, 'Very Good': 4, 'Good': 3, 'Fair': 2, 'Poor': 1}
        condition_score = condition_scores.get(condition, 3)
        
        # Create input DataFrame
        input_dict = {
            'make': [make],
            'model': [model_name],
            'year': [year],
            'mileage': [mileage],
            'condition': [condition],
            'body_type': [body_type],
            'fuel_type': [fuel_type],
            'transmission': [transmission],
            'owners': [owners],
            'engine_size': [engine_size],
            'age': [age],
            'condition_score': [condition_score],
            'service_score': [2]
        }
        
        input_df = pd.DataFrame(input_dict)
        
        # Encode categorical variables
        for col in categorical_cols:
            if col in input_df.columns and col in label_encoders:
                try:
                    input_df[col] = label_encoders[col].transform(input_df[col])
                except Exception as e:
                    logger.warning(f"Failed to encode {col}: {e}, using 0")
                    input_df[col] = 0
        
        # Scale numerical features
        if scaler is not None and numerical_cols is not None:
            for col in numerical_cols:
                if col in input_df.columns:
                    values = input_df[col].values.reshape(-1, 1)
                    input_df[col] = scaler.transform(values)
        
        # Ensure correct column order
        if features is not None:
            # Add missing columns with 0
            for col in features:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[features]
        
        # Predict
        prediction = model.predict(input_df)[0]
        prediction = round(prediction / 100) * 100
        
        result = {
            'success': True,
            'predicted_value': float(prediction),
            'model': 'ml_randomforest_v1',
            'message': 'Prediction successful'
        }
        
        logger.info(f"Prediction result: ${prediction}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Prediction failed'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'service': 'tradein_valuation_v1',
        'models_loaded': {
            'model': model is not None,
            'scaler': scaler is not None,
            'features': features is not None,
            'encoders': {k: v is not None for k, v in label_encoders.items()}
        }
    })

@app.route('/features', methods=['GET'])
def get_features():
    """Returns list of features used by the model"""
    return jsonify({
        'success': True,
        'features': {
            'categorical': categorical_cols,
            'numerical': numerical_cols if numerical_cols else [],
            'total_features': len(features) if features is not None else 0
        }
    })

if __name__ == '__main__':
    # Load models
    model_loaded = load_models()
    
    port = int(os.environ.get('PORT', 5003))
    print(f"\nStarting trade-in valuation server on port {port}")
    print(f"Model loaded: {model_loaded}")
    print("Server is ready! Waiting for requests...")
    print(f"\nTest the server:")
    print(f"  Health check: http://localhost:{port}/health")
    print(f"  Features: http://localhost:{port}/features")
    print(f"  Predict endpoint: POST http://localhost:{port}/predict")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=True)